# Godard

import numpy as np
import tensorflow as tf
np.random.seed(317)
tf.set_random_seed(318)
rng = np.random

from glob import glob
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, LambdaCallback
from keras.layers.advanced_activations import PReLU
from math import ceil
from os import path, mkdir, listdir
from PIL import Image
from sklearn.model_selection import train_test_split
from time import time
import logging
import keras.backend as K
import pandas as pd
import pdb
import tifffile as tif


import sys
sys.path.append('.')
from planet.utils.data_utils import tagstr_to_binary, TAGS, TAGS_short
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


def rename(newname):
    '''Decorator for procedurally generating functions with different names.'''
    def decorator(f):
        f.__name__ = newname
        return f
    return decorator


class Godard(object):

    def __init__(self, checkpoint_name='godard_jpg_softmax'):

        self.config = {
            'input_shape': [64, 64, 3],
            'output_shape': [17],
            'batch_size_tst': 2000,
            'batch_size_trn': 64,
            'nb_epochs': 300,
            'nb_augment_max': 10,
            'prop_val': 0.2,
            'imgs_csv_trn': 'data/train_v2.csv',
            'imgs_dir_trn': 'data/train-jpg',
            'imgs_csv_tst': 'data/sample_submission_v2.csv',
            'imgs_dir_tst': 'data/test-jpg',
            'imgs_ext': 'jpg',
            'cpname': checkpoint_name
        }

        self.net = None

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.config['cpname'], '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self, weights_path=None):

        logger = logging.getLogger(funcname())

        inputs = Input(shape=self.config['input_shape'])
        x = inputs

        def conv_block(nb_filters, prop_dropout, x):
            ki = 'he_uniform'
            x = Conv2D(nb_filters, (3, 3), padding='same', kernel_initializer=ki)(x)
            x = PReLU()(x)
            x = Conv2D(nb_filters, (3, 3), padding='same', kernel_initializer=ki)(x)
            x = PReLU()(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = Dropout(prop_dropout)(x)
            return x

        x = conv_block(32, 0.1, x)
        x = conv_block(64, 0.2, x)
        x = conv_block(128, 0.3, x)
        x = conv_block(256, 0.3, x)
        x = conv_block(512, 0.3, x)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = PReLU()(x)
        x = Dropout(0.2)(x)
        shared_out = x

        # Individual softmax classifier for each class.
        classifiers = []
        for _ in range(17):
            x = Dense(256)(shared_out)
            x = PReLU()(x)
            x = Dense(2, activation='softmax', name='output_%d' % _)(x)
            classifiers.append(x)

        x = concatenate(classifiers, axis=-1)
        x = Reshape((17, 2))(x)
        x = Lambda(lambda x: x[:, :, 1])(x)

        self.net = Model(inputs=inputs, outputs=x)

        def myt(yt, yp):
            return K.mean(yt)

        def myp(yt, yp):
            return K.mean(yp)

        def prec(yt, yp):
            yp = K.round(yp)
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            return tp / (tp + fp + K.epsilon())

        def reca(yt, yp):
            yp = K.round(yp)
            tp = K.sum(yt * yp)
            fn = K.sum(K.clip(yt - yp, 0, 1))
            return tp / (tp + fn + K.epsilon())

        def F2(yt, yp):
            p = prec(yt, yp)
            r = reca(yt, yp)
            b = 2.0
            return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))

        def logloss(yt, yp):
            wfn = 6.  # Weight false negative errors.
            wfp = 1.  # Weight false positive errors.
            wmult = (yt * wfn) + (K.abs(yt - 1) * wfp)
            errors = yt * K.log(yp + 1e-7) + ((1 - yt) * K.log(1 - yp + 1e-7))
            return -1 * K.mean(errors * wmult)

        # Generate an F2 metric for each tag.
        tf2_metrics = []
        for i, t in enumerate(TAGS_short):
            @rename('F2_%s' % t)
            def tagF2(yt, yp, i=i):
                return F2(yt[:, i], yp[:, i])
            tf2_metrics.append(tagF2)

        # Generate a metric for each tag that tracks how often it occurs in a batch.
        tcnt_metrics = []
        for i, t in enumerate(TAGS_short):
            @rename('cnt_%s' % t)
            def tagcnt(yt, yp, i=i):
                return K.sum(yt[:, i])
            tcnt_metrics.append(tagcnt)

        self.net.compile(optimizer=Adam(0.002),
                         metrics=[F2, prec, reca, myt, myp] + tf2_metrics + tcnt_metrics,
                         loss=logloss)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        if weights_path is not None:
            logger.info('Loading weights from %s' % weights_path)
            self.net.load_weights(weights_path)

    def train(self):

        logger = logging.getLogger(funcname())

        # Data setup.
        iidxs = np.arange(len(listdir(self.config['imgs_dir_trn'])))
        iidxs_trn, iidxs_val = train_test_split(iidxs, test_size=self.config['prop_val'], random_state=rng)
        steps_trn = ceil(len(iidxs_trn) / self.config['batch_size_trn'])
        steps_val = ceil(len(iidxs_val) / self.config['batch_size_trn'])
        assert len(set(iidxs_trn).intersection(iidxs_val)) == 0
        assert steps_val < steps_trn

        gen_trn = self.batch_gen(iidxs_trn, steps_trn, nb_augment_max=self.config['nb_augment_max'])
        gen_val = self.batch_gen(iidxs_val, steps_val, nb_augment_max=1)

        def print_tag_F2_metrics(epoch, logs):

            for tag in TAGS_short:
                f2_trn = logs['F2_%s' % tag]
                f2_val = logs['val_F2_%s' % tag]
                cnt_trn = logs['cnt_%s' % tag]
                cnt_val = logs['val_cnt_%s' % tag]
                logger.info('%-6s F2 trn=%-6.3lf cnt=%-6.2lf F2 val=%-6.3lf cnt=%-6.2lf' %
                            (tag, f2_trn, cnt_trn, f2_val, cnt_val))

        cb = [
            LambdaCallback(on_epoch_end=print_tag_F2_metrics),
            HistoryPlot('%s/history.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalF2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='val_F2', factor=0.5, patience=10,
                              min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max'),
            EarlyStopping(monitor='val_F2', patience=30, verbose=1, mode='max')
        ]

        self.net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.config['nb_epochs'],
                               verbose=1, callbacks=cb,
                               workers=3, pickle_safe=True,
                               validation_data=gen_val, validation_steps=steps_val)

    def batch_gen(self, iidxs, nb_steps=100, nb_augment_max=0):

        # Constants.
        ibatch_shape = [self.config['batch_size_trn'], ] + self.config['input_shape']
        tbatch_shape = [self.config['batch_size_trn'], ] + self.config['output_shape']

        # Read the CSV and extract image paths and tags.
        df = pd.read_csv(self.config['imgs_csv_trn'])
        paths = ['%s/%s.jpg' % (self.config['imgs_dir_trn'], n) for n in df['image_name'].values]
        tags = [tagstr_to_binary(ts) for ts in df['tags'].values]

        aug_funcs = [
            lambda x: x,
            lambda x: np.rot90(x, k=rng.randint(1, 4), axes=(0, 1)),
            lambda x: np.flipud(x),
            lambda x: np.fliplr(x)
        ]

        while True:

            # Shuffled cycle image indexes for consecutive sampling.
            rng.shuffle(iidxs)
            iidxs_cycle = cycle(iidxs)
            iidxs_sampled = []

            for _ in range(nb_steps):

                ibatch = np.zeros(ibatch_shape, dtype=np.float32)
                tbatch = np.zeros(tbatch_shape, dtype=np.uint8)

                for bidx in range(self.config['batch_size_trn']):
                    iidx = next(iidxs_cycle)
                    ibatch[bidx] = self.img_path_to_img(paths[iidx])
                    tbatch[bidx] = tags[iidx]
                    for aug in rng.choice(aug_funcs, nb_augment_max):
                        ibatch[bidx] = aug(ibatch[bidx])
                    iidxs_sampled.append(iidx)

                yield ibatch, tbatch

            assert set(iidxs_sampled) == set(iidxs)

    def img_path_to_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img.thumbnail(self.config['input_shape'][:2])
        img = np.asarray(img)
        return img / 255.

    def predict(self, imgs_names):

        imgs_dir = self.config['trn_imgs_dir'] if 'train' in imgs_names[0] else self.config['tst_imgs_dir']
        imgs_paths = ['%s/%s.%s' % (imgs_dir, name, self.config['img_ext']) for name in imgs_names]
        shape = [len(imgs_names), ] + self.config['input_shape']
        imgs_batch = np.zeros(shape, dtype=np.float32)

        for bidx, img_path in enumerate(imgs_paths):
            imgs_batch[bidx] = self.img_path_to_img(img_path)

        tags_pred = self.net.predict(imgs_batch)
        for i in range(len(tags_pred)):
            tags_pred[i] = correct_tags(tags_pred[i])

        return tags_pred.round().astype(np.uint8)

    def visualize_activation(self):

        from matplotlib import pyplot as plt
        from vis.utils import utils
        from vis.visualization import visualize_activation, get_num_filters

        vis_images = []
        for i in range(self.config['output_shape'][0]):
            # The name of the layer we want to visualize
            layer_name = 'output_%d' % i
            layer_idx = [idx for idx, layer in enumerate(self.net.layers) if layer.name == layer_name][0]

            print('Working on %s' % layer_name)
            # Generate input image for each filter. Here `text` field is used to
            # overlay `filter_value` on top of the image.
            for idx in [1, 1, 1]:
                img = visualize_activation(self.net, layer_idx, filter_indices=idx, max_iter=500)
                img = utils.draw_text(img, TAGS[i])
                vis_images.append(img)

        # Generate stitched image palette with 8 cols.
        stitched = utils.stitch_images(vis_images, cols=3)
        plt.axis('off')
        plt.imshow(stitched)
        plt.title(self.checkpoint_name)
        plt.show()

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(Godard())
