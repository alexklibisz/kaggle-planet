# ResNet50

import numpy as np
import tensorflow as tf
np.random.seed(317)
tf.set_random_seed(318)

from glob import glob
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, LambdaCallback
from keras.losses import kullback_leibler_divergence as KLD
from keras.applications import resnet50
from math import ceil
from os import path, mkdir, listdir
from PIL import Image
from time import time
import argparse
import logging
import keras.backend as K
import pandas as pd
import tifffile as tif
from scipy.misc import imread

import sys
sys.path.append('.')
from planet.utils.data_utils import tagstr_to_ints, onehot_to_taglist, TAGS, TAGS_short, onehot_F2, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


class ResNet50_softmax(object):

    def __init__(self, checkpoint_name='ResNet50_softmax'):
        self.config = {
            'input_shape': [224, 224, 3],
            'output_shape': [17],
            'batch_size_tst': 150,
            'batch_size_trn': 32,
            'trn_nb_epochs': 300,
            'trn_transform': True,
            'trn_imgs_csv': 'data/train_v2.csv',
            'trn_imgs_dir': 'data/train-jpg',
            'tst_imgs_csv': 'data/sample_submission_v2.csv',
            'tst_imgs_dir': 'data/test-jpg',
            'extension': 'jpg',
            'trn_prop_trn': 0.8,
            'trn_prop_val': 0.2
        }

        self.checkpoint_name = checkpoint_name
        self.imgs = []
        self.lbls = []
        self.net = None

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.checkpoint_name, '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self, weights_path=None):

        # ResNet50 setup as a big feature extractor.
        inputs = Input(shape=self.config['input_shape'])
        res = resnet50.ResNet50(include_top=False, input_tensor=inputs, weights='imagenet')

        # Add a classifier for each class instead of a single classifier.
        x = Flatten()(res.output)
        x = PReLU()(x)
        shared_out = Dropout(0.3)(x)
        classifiers = []
        for n in range(17):
            classifiers.append(Dense(2, activation='softmax')(shared_out))

        # Concatenate classifiers and reshape to match output shape.
        x = concatenate(classifiers, axis=-1)
        x = Reshape((17, 2))(x)
        x = Lambda(lambda x: x[:, :, 1])(x)

        self.net = Model(inputs=res.input, outputs=x)

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
            wfn = 8.  # Weight false negative errors.
            wfp = 1.  # Weight false positive errors.
            wmult = (yt * wfn) + (K.abs(yt - 1) * wfp)
            errors = yt * K.log(yp + 1e-7) + ((1 - yt) * K.log(1 - yp + 1e-7))
            return -1 * K.mean(errors * wmult)

        # Decorator for procedurally generating functions with different names.
        def rename(newname):
            def decorator(f):
                f.__name__ = newname
                return f
            return decorator

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

        self.net.compile(optimizer=Adam(0.001), metrics=[F2, prec, reca] + tf2_metrics + tcnt_metrics, loss=logloss)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        if weights_path is not None:
            self.net.load_weights(weights_path)

    def train(self):

        logger = logging.getLogger(funcname())
        rng = np.random

        imgs_idxs = np.arange(len(listdir(self.config['trn_imgs_dir'])))
        imgs_idxs = rng.choice(imgs_idxs, len(imgs_idxs))
        imgs_idxs_trn = imgs_idxs[:int(len(imgs_idxs) * self.config['trn_prop_trn'])]
        imgs_idxs_val = imgs_idxs[-int(len(imgs_idxs) * self.config['trn_prop_val']):]
        gen_trn = self.batch_gen(self.config['trn_imgs_csv'], self.config['trn_imgs_dir'], imgs_idxs_trn,
                                 transform=self.config['trn_transform'], balanced=True)
        gen_val = self.batch_gen(self.config['trn_imgs_csv'], self.config['trn_imgs_dir'], imgs_idxs_val, False)

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
            ModelCheckpoint('%s/weights_val_F2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='val_F2', factor=0.5, patience=2,
                              min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max'),
            EarlyStopping(monitor='val_F2', patience=30, verbose=1, mode='max')
        ]

        nb_steps_trn = ceil(len(imgs_idxs_trn) * 1. / self.config['batch_size_trn'])
        nb_steps_val = ceil(len(imgs_idxs_val) * 1. / self.config['batch_size_trn'])

        self.net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=self.config['trn_nb_epochs'],
                               verbose=1, callbacks=cb,
                               workers=3, pickle_safe=True,
                               validation_data=gen_val, validation_steps=nb_steps_val)

    def batch_gen(self, imgs_csv, imgs_dir, img_idxs, transform=False, balanced=False):

        # Helpers.
        rng = np.random
        shuffle = lambda x: rng.choice(x, len(x))

        # Constants.
        nb_steps = ceil(len(img_idxs) / self.config['batch_size_trn'])
        img_batch_shape = [self.config['batch_size_trn'], ] + self.config['input_shape']
        tag_batch_shape = [self.config['batch_size_trn'], ] + self.config['output_shape']

        # Read the CSV and extract image paths and tags.
        df = pd.read_csv(imgs_csv)
        img_pths = ['%s/%s.jpg' % (imgs_dir, n) for n in df['image_name'].values]
        img_tags = [tagstr_to_ints(tstr) for tstr in df['tags'].values]

        # Mapping from tag to image indexes.
        tags_to_img_idxs = [[] for _ in range(len(TAGS))]
        for img_idx, tags in enumerate(img_tags):
            pos_tags, = np.where(tags == 1.)
            for t in pos_tags:
                tags_to_img_idxs[t].append(img_idx)

        # Convert to cycles for sequential sampling.
        for i in range(len(TAGS)):
            tags_to_img_idxs[i] = cycle(shuffle(tags_to_img_idxs[i]))

        # Cycle on image indexes for consecutive sampling (when balanced = False).
        img_idxs_cycle = cycle(img_idxs)

        while True:

            # Shuffle order of tags for balanced sampling.
            tag_idxs = cycle(shuffle(np.arange(len(TAGS))))

            for _ in range(nb_steps):

                imgs_batch = np.zeros(img_batch_shape, dtype=np.float32)
                tags_batch = np.zeros(tag_batch_shape, dtype=np.uint8)

                for batch_idx in range(self.config['batch_size_trn']):

                    if balanced:
                        tag_idx = next(tag_idxs)
                        img_idx = next(tags_to_img_idxs[tag_idx])
                    else:
                        img_idx = next(img_idxs_cycle)

                    imgs_batch[batch_idx] = self.img_path_to_img(img_pths[img_idx])
                    tags_batch[batch_idx] = img_tags[img_idx]

                    if transform:
                        imgs_batch[batch_idx] = random_transforms(imgs_batch[batch_idx], 0, 4)

                yield imgs_batch, tags_batch

    def img_path_to_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img.thumbnail(self.config['input_shape'][:2])
        img = np.asarray(img, dtype=np.float32) / 255.
        return img

    def predict(self, imgs_names):

        imgs_dir = self.config['trn_imgs_dir'] if 'train' in imgs_names[0] else self.config['tst_imgs_dir']
        imgs_paths = ['%s/%s.%s' % (imgs_dir, name, self.config['extension']) for name in imgs_names]
        shape = [len(imgs_names), ] + self.config['input_shape']
        imgs_batch = np.zeros(shape, dtype=np.float32)

        for bidx, img_path in enumerate(imgs_paths):
            imgs_batch[bidx] = self.img_path_to_img(img_path)

        return self.net.predict(imgs_batch).round().astype(np.uint8)

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(ResNet50_softmax())
