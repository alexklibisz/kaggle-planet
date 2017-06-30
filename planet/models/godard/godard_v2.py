# Godard is back.

import numpy as np
import tensorflow as tf
np.random.seed(317)
tf.set_random_seed(318)
rng = np.random

from glob import glob
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, LambdaCallback
from keras.layers.advanced_activations import PReLU, ELU
from math import ceil
from os import path, mkdir, listdir, environ
from PIL import Image
from skimage.transform import resize, rotate
from sklearn.model_selection import train_test_split
from time import time
import logging
import keras.backend as K
import pandas as pd
import pickle as pkl
import pdb
import tifffile as tif


import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, TAGS_short, IMG_MEAN_JPG_TRN, tagstr_to_binary, correct_tags, tag_proportions
from planet.utils.keras_utils import HistoryPlotCB, ValidationCB, ParamStatsCB, prec, reca, F2
from planet.utils.runtime import funcname


class Godard(object):

    def __init__(self, checkpoint_name='godard_v2_b'):

        self.config = {
            'output_shape': (17,),
            'batch_size_tst': 2000,
            'batch_size_trn': 64,
            'nb_epochs': 300,
            'nb_augment_max': 10,
            'cpname': checkpoint_name,

            # JPG config.
            'input_shape': (150, 150, 3),
            'imgs_csv_trn': 'data/train_v2.csv',
            'imgs_dir_trn': 'data/train-jpg',
            'imgs_csv_tst': 'data/sample_submission_v2.csv',
            'imgs_dir_tst': 'data/test-jpg',
            'img_ext': 'jpg',
            'cache_imgs': False,

            'trn_idxs_path': 'data/idxs_trn.pkl',
            'val_idxs_path': 'data/idxs_val.pkl'
        }

        self.net = None
        self.imgs_cache = {}

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.config['cpname'], '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self, weights_path=None):

        logger = logging.getLogger(funcname())
        cki = 'he_normal'
        dki = 'glorot_normal'

        inputs = Input(shape=self.config['input_shape'])

        # 2x Conv-batchnorm-prelu-dropout, maxpool
        def conv_block(nb_filters, prop_dropout, x):
            x = Conv2D(nb_filters, 3, padding='same', kernel_initializer=cki, name='conv_%d_0' % nb_filters)(x)
            x = BatchNormalization(momentum=0.5)(x)
            x = ELU()(x)
            x = Dropout(prop_dropout)(x)
            x = Conv2D(nb_filters, 3, padding='same', kernel_initializer=cki)(x)
            x = MaxPooling2D(pool_size=2)(x)
            x = BatchNormalization(momentum=0.5)(x)
            x = ELU()(x)
            x = Dropout(prop_dropout)(x)
            return x

        # Conv/pooling layers.
        x = conv_block(32, 0.1, inputs)
        x = conv_block(64, 0.1, x)
        x = conv_block(128, 0.2, x)
        x = conv_block(256, 0.2, x)
        x = conv_block(512, 0.2, x)

        # Shared fully-connected layers.
        # Dense-batchnorm-ELU-dropout
        x = Flatten()(x)
        x = Dense(1024, name='dense_shared_0', kernel_initializer=dki)(x)
        x = BatchNormalization(momentum=0.5)(x)
        x = ELU()(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, name='dense_shared_1', kernel_initializer=dki)(x)
        x = BatchNormalization(momentum=0.5)(x)
        x = ELU()(x)
        x = Dropout(0.2)(x)
        shared = x

        # Single softmax classifier for the four cloud-cover tags.
        # Dense-batchnorm-ELU-dense-batchnorm-classification
        x = Dense(256, name='dense_clouds_0')(shared)
        x = BatchNormalization(momentum=0.5)(x)
        x = ELU()(x)
        x = Dense(4, name='dense_clouds_1', kernel_initializer=dki)(x)
        x = BatchNormalization(momentum=0.5)(x)
        out_clds = Activation('softmax', name='out_clds')(x)

        # Single sigmoid classifier for remaining tags.
        # Dense-batchnorm-ELU-dense-batchnorm-classification
        x = Dense(256, name='dense_rest_0', kernel_initializer=dki)(shared)
        x = ELU()(x)
        x = Dense(13, name='dense_rest_1', kernel_initializer=dki)(x)
        x = BatchNormalization(momentum=0.5)(x)
        out_rest = Activation('sigmoid', name='out_rest')(x)

        # Some surgery to combine outputs into one tensor and re-arrange them to
        # match the alphabetical tag ordering.
        TAGS_net = ['clear', 'cloudy', 'haze', 'partly_cloudy', 'agriculture', 'artisinal_mine',
                    'bare_ground', 'blooming', 'blow_down', 'conventional_mine', 'cultivation',
                    'habitation', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']
        concat = concatenate([out_clds, out_rest])
        arranged = [Lambda(lambda x: K.expand_dims(x[:, TAGS_net.index(t)]))(concat) for t in TAGS]
        out = concatenate(arranged, name='out_arranged')

        self.net = Model(inputs=inputs, outputs=out)

        # Most tags in the dataset have a strong class imbalance (way more
        # negatives than positives or vice versa). With regular log-loss, there is
        # incentive to predict the dominant class. To compensate for this, we
        # up-weight the error for the non-dominant class and down-weight the error
        # for the dominant class. This increases penalty for ignoring the
        # non-dominant class and increases penalty for defaulting to the dominant
        # class - approximating a 50/50 class balance.
        ppos, pneg = tag_proportions()
        wpos = np.ones_like(ppos) * 0.5 / ppos
        wneg = np.ones_like(pneg) * 0.5 / pneg

        def loss(yt, yp, wpos=K.variable(wpos), wneg=K.variable(wneg)):
            '''Log loss per tag per data point (b, c). Then compute and apply weight matrix.'''
            loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))  # (b, c)
            wmat = (yt * wpos) + (K.abs(yt - 1) * wneg)     # (b,c)
            return K.mean(loss * wmat, axis=-1)             # (b,1)

        self.net.compile(optimizer=Adam(0.001), metrics=[F2, prec, reca, 'binary_crossentropy'], loss=loss)
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        if weights_path is not None:
            logger.info('Loading weights from %s' % weights_path)
            self.net.load_weights(weights_path)

    # def train(self):

    #     logger = logging.getLogger(funcname())

    #     with open(self.config['trn_idxs_path'], 'rb') as f:
    #         idxs_trn_ = pkl.load(f)

    #     nbs = [10, 100, 500, 1000, 2500, 5000]
    #     for nb in nbs:
    #         self.net = None
    #         self.create_net()
    #         idxs_trn = rng.choice(idxs_trn_, nb)
    #         steps_trn = ceil(len(idxs_trn) / self.config['batch_size_trn'])
    #         gen_trn = self.batch_gen(idxs_trn, steps_trn, nb_augment_max=self.config['nb_augment_max'])
    #         epochs = 100
    #         history = self.net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=epochs, verbose=1)

    #         with open('out.txt', 'a') as f:
    #             s = 'nb = %d, max (f2, prec, reca) = (%.3lf,%.3lf,%.3lf)' % (nb, np.max(history.history['F2']),
    #                                                                          np.max(history.history['prec']),
    #                                                                          np.max(history.history['reca']))
    #             f.write(s + '\n')
    #             print(s)

    def train(self):

        logger = logging.getLogger(funcname())

        # Data setup.
        idxs_trn = pkl.load(open(self.config['trn_idxs_path'], 'rb'))
        idxs_val = pkl.load(open(self.config['val_idxs_path'], 'rb'))
        assert set(idxs_trn).intersection(idxs_val) == set([])
        steps_trn = ceil(len(idxs_trn) / self.config['batch_size_trn'])
        steps_val = ceil(len(idxs_val) / self.config['batch_size_trn'])
        gen_trn = self.batch_gen(idxs_trn, steps_trn, nb_augment_max=self.config['nb_augment_max'])
        gen_val = self.batch_gen(idxs_val, steps_val, nb_augment_max=0)

        cb = [
            ValidationCB(self.cpdir, gen_val, self.config['batch_size_trn'], steps_val),
            ParamStatsCB(self.cpdir),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalF2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='val_F2', factor=0.75, patience=10,
                              min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max'),
            EarlyStopping(monitor='val_F2', patience=30, verbose=1, mode='max')
        ]

        self.net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.config['nb_epochs'],
                               verbose=1, callbacks=cb)

    def batch_gen(self, idxs, nb_steps=100, nb_augment_max=0):

        # Constants.
        ibatch_shape = (self.config['batch_size_trn'],) + self.config['input_shape']
        tbatch_shape = (self.config['batch_size_trn'],) + self.config['output_shape']

        # Read the CSV and extract image paths and tags.
        df = pd.read_csv(self.config['imgs_csv_trn'])
        paths = ['%s/%s.%s' % (self.config['imgs_dir_trn'], n, self.config['img_ext']) for n in df['image_name'].values]
        tags = [tagstr_to_binary(ts) for ts in df['tags'].values]

        aug_funcs = [
            lambda x: x,
            lambda x: np.flipud(x),
            lambda x: np.fliplr(x),
            lambda x: rotate(x, rng.randint(0, 360), mode='reflect'),
        ]

        while True:

            # Shuffled cycle image indexes for consecutive sampling.
            rng.shuffle(idxs)
            idxs_cycle = cycle(idxs)

            for _ in range(nb_steps):

                ibatch = np.zeros(ibatch_shape, dtype=np.float32)
                tbatch = np.zeros(tbatch_shape, dtype=np.uint8)

                for bidx in range(self.config['batch_size_trn']):
                    iidx = next(idxs_cycle)
                    ibatch[bidx] = self.get_img(paths[iidx], cache=self.config['cache_imgs'])
                    tbatch[bidx] = tags[iidx]
                    for aug in rng.choice(aug_funcs, nb_augment_max):
                        ibatch[bidx] = aug(ibatch[bidx])

                yield ibatch, tbatch

    def predict(self, imgs_names):

        imgs_dir = self.config['imgs_dir_trn'] if 'train' in imgs_names[0] else self.config['imgs_dir_tst']
        imgs_paths = ['%s/%s.%s' % (imgs_dir, name, self.config['img_ext']) for name in imgs_names]
        shape = (len(imgs_names), ) + self.config['input_shape']
        imgs_batch = np.zeros(shape, dtype=np.float32)

        for bidx, img_path in enumerate(imgs_paths):
            imgs_batch[bidx] = self.get_img(img_path)

        tags_pred = self.net.predict(imgs_batch)
        for i in range(len(tags_pred)):
            tags_pred[i] = correct_tags(tags_pred[i])

        return tags_pred.round().astype(np.uint8)

    def get_img(self, img_path, cache=False):
        '''Optionally cache the images as uint8 arrays to save on memory.
        Then pre-process each image after retrieving from cache or disk.'''

        if cache and img_path in self.imgs_cache:
            img = self.imgs_cache[img_path]
        else:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail(self.config['input_shape'][:2])
            img = np.asarray(img, dtype=np.uint8)

        if cache and img_path not in self.imgs_cache:
            self.imgs_cache[img_path] = img

        # Scale to [0,1] and subtract mean per channel.
        img = img.astype(np.float32) / 255.
        for c in range(img.shape[-1]):
            img[:, :, c] -= IMG_MEAN_JPG_TRN[c] / 255.

        return img

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
                # img = utils.draw_text(img, TAGS[i])
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
