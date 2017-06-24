# Godard

import numpy as np
import tensorflow as tf
np.random.seed(317)
tf.set_random_seed(318)

from glob import glob
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, LearningRateScheduler
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from math import ceil
from os import path, mkdir, listdir
from PIL import Image
from skimage.transform import resize
from time import time
import argparse
import logging
import keras.backend as K
import pandas as pd
import pdb
import tifffile as tif


import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_ints, onehot_to_taglist, TAGS, onehot_F2, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


class Godard(object):

    def __init__(self, checkpoint_name='godard_jpg_softmax'):

        self.config = {
            'input_shape': [64, 64, 3],
            'output_shape': [17],
            'batch_size_tst': 1500,
            'batch_size_trn': 64,
            'trn_nb_epochs': 120,
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
        self.net = None
        self.rng = np.random

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.checkpoint_name, '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self, weights_path=None):

        inputs = Input(shape=self.config['input_shape'])

        x = BatchNormalization(axis=3)(inputs)

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
            x = Dense(2, activation='softmax')(x)
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
            wfn = 8.  # Weight false negative errors.
            wfp = 1.  # Weight false positive errors.
            wmult = (yt * wfn) + (K.abs(yt - 1) * wfp)
            errors = yt * K.log(yp + 1e-7) + ((1 - yt) * K.log(1 - yp + 1e-7))
            return -1 * K.mean(errors * wmult)

        def acc(yt, yp):
            nbwrong = K.sum(K.abs(yt - K.round(yp)))
            size = K.sum(K.ones_like(yt))
            return (size - nbwrong) / size

        self.net.compile(optimizer=Adam(0.0015), metrics=[F2, prec, reca, myt, myp, acc], loss=logloss)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        if weights_path is not None:
            self.net.load_weights(weights_path)

    def train(self, rng=np.random):

        imgs_idxs = np.arange(len(listdir(self.config['trn_imgs_dir'])))
        imgs_idxs = rng.choice(imgs_idxs, len(imgs_idxs))
        imgs_idxs_trn = imgs_idxs[:int(len(imgs_idxs) * self.config['trn_prop_trn'])]
        imgs_idxs_val = imgs_idxs[-int(len(imgs_idxs) * self.config['trn_prop_val']):]
        gen_trn = self.batch_gen(self.config['trn_imgs_csv'], self.config['trn_imgs_dir'], imgs_idxs_trn,
                                 self.config['trn_transform'])
        gen_val = self.batch_gen(self.config['trn_imgs_csv'], self.config['trn_imgs_dir'], imgs_idxs_val, False)

        cb = [
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

    def batch_gen(self, imgs_csv, imgs_dir, imgs_idxs, transform=False):

        rng = np.random
        nb_steps = ceil(len(imgs_idxs) * 1. / self.config['batch_size_trn'])

        # Read the CSV and extract image names and tags.
        df = pd.read_csv(imgs_csv)
        imgs_paths = ['%s/%s.jpg' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.strip().split(' ')) for t in df['tags'].values]

        while True:

            # Shuffle all the given images and make one complete pass through them before re-shuffling.
            _imgs_idxs = cycle(rng.choice(imgs_idxs, len(imgs_idxs)))
            for _ in range(nb_steps):

                imgs_batch = np.zeros([self.config['batch_size_trn'], ] + self.config['input_shape'])
                tags_batch = np.zeros([self.config['batch_size_trn'], ] + self.config['output_shape'])

                for batch_idx in range(self.config['batch_size_trn']):
                    img_idx = next(_imgs_idxs)
                    img = self.img_path_to_img(imgs_paths[img_idx])
                    if transform:
                        img = random_transforms(img, nb_min=0, nb_max=4)
                    imgs_batch[batch_idx] = img
                    tags_batch[batch_idx] = tagset_to_ints(tag_sets[img_idx])

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
    model_runner(Godard())
