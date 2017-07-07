# Fully-convolutional autoencoder learns to reconstruct all of the images.
# The bottleneck layer is used as a feature vector to perform KNN lookup for images.
# The nearest-neighbors' tag vectors are used to determine the image's tag vector.
# The key assumption is that the bottleneck layer is a representative of the image's contents.

from itertools import cycle
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape, Input, UpSampling2D
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN, Callback
from keras.regularizers import l2
from pprint import pprint
from time import time, sleep
import keras.backend as K
import json
import h5py
import logging
import math
import numpy as np
import os

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, serialize_config, IMG_MEAN_JPG_TRN
from planet.utils.keras_utils import TensorBoardWrapper, HistoryPlotCB
rng = np.random


def _autoencoder(input_shape=(256, 256, 3)):

    def preprocess_jpg(x):
        x[:, :, :, 0] -= IMG_MEAN_JPG_TRN[0]
        x[:, :, :, 1] -= IMG_MEAN_JPG_TRN[1]
        x[:, :, :, 2] -= IMG_MEAN_JPG_TRN[2]
        return x / 255.

    input = Input(shape=input_shape)

    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = MaxPooling2D(2, strides=2)(x)
    x = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    enc = MaxPooling2D(2, name='encoder_out')(x)

    x = UpSampling2D(2)(enc)
    x = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = UpSampling2D(2)(x)
    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    dec = Conv2D(3, 3, activation='tanh', padding='same', kernel_initializer='he_normal')(x)

    autoencoder = Model(input, dec)
    return autoencoder, preprocess_jpg


class ExamplesCB(Callback):

    def __init__(self, cpdir, batch_gen):
        super().__init__()
        self.cpdir = cpdir
        self.batch_gen = batch_gen

    def on_batch_end(self, b, l):
        imgs, _ = next(self.batch_gen)[:10]
        print(imgs.shape)
        sys.exit(0)

    def on_epoch_end(self, epoch, logs):

        from scipy.misc import imsave

        imgs, _ = next(self.batch_gen)[:10]
        prds = self.model.predict(imgs)

        for i, (img, prd) in enumerate(zip(imgs, prds)):
            p = '%s/example_%03d_%02d' % (self.cpdir, epoch, i)
            imsave(p, np.hstack(img, prd))


class AutoKNN(object):

    def __init__(self):

        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/autoknn_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (96, 96, 3),
            'preprocess_imgs_func': lambda x: x,

            # Network setup.
            'net_builder_func': _autoencoder,
            'net_loss_func': 'mse',

            # Training setup.
            'trn_epochs': 100,
            'trn_augment_max_trn': 2,
            'trn_augment_max_val': 1,
            'trn_batch_size': 48,
            'trn_optimizer': Adam,
            'trn_optimizer_args': {'lr': 0.01},
            'trn_prop_data': 0.5,

            # Testing.
            'tst_batch_size': 1000
        }

    @property
    def cpdir(self):
        if not os.path.exists(self.cfg['cpdir']):
            os.mkdir(self.cfg['cpdir'])
        return self.cfg['cpdir']

    def train(self, weights_path=None, callbacks=[]):

        # Network setup. ResNet layers frozen at first.
        net, self.cfg['preprocess_imgs_func'] = self.cfg['net_builder_func'](self.cfg['input_shape'])
        json.dump(net.to_json(), open('%s/model.json' % self.cpdir, 'w'), indent=2)
        json.dump(serialize_config(self.cfg), open('%s/config.json' % self.cpdir, 'w'))

        data_trn = h5py.File(self.cfg['hdf5_path_trn'])
        data_tst = h5py.File(self.cfg['hdf5_path_tst'])
        nb_examples = len(data_trn.get('images')) + len(data_tst.get('images'))
        steps_trn = math.ceil(nb_examples / self.cfg['trn_batch_size'])

        steps_trn = 100
        gen_trn = self.batch_gen(steps_trn)

        opt = self.cfg['trn_optimizer'](**self.cfg['trn_optimizer_args'])
        net.compile(optimizer=opt, loss=self.cfg['net_loss_func'])

        net.summary()
        if weights_path is not None:
            net.load_weights(weights_path)
        pprint(self.cfg)

        cb = [
            ExamplesCB(self.cfg['cpdir'], gen_trn),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            EarlyStopping(monitor='loss', min_delta=0.01, patience=20, verbose=1, mode='max'),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wloss.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            TerminateOnNaN()
        ]

        train = net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.cfg['trn_epochs'],
                                  verbose=1, callbacks=cb)

        return train.history

    def batch_gen(self, nb_steps):

        data_trn = h5py.File(self.cfg['hdf5_path_trn'], 'r')
        imgs_trn = data_trn.get('images')
        data_tst = h5py.File(self.cfg['hdf5_path_tst'], 'r')
        imgs_tst = data_tst.get('images')
        ib_shape = (self.cfg['trn_batch_size'],) + self.cfg['input_shape']
        tb_shape = (self.cfg['trn_batch_size'], len(TAGS))
        didxs = np.arange(len(imgs_trn) + len(imgs_tst))

        def crop(img):
            h, w, _ = self.cfg['input_shape']
            y0 = rng.randint(0, img.shape[0] - h)
            x0 = rng.randint(0, img.shape[1] - w)
            y1 = y0 + h
            x1 = x0 + w
            return img[y0:y1, x0:x1, :]

        def get_img(idx):
            if idx < len(imgs_trn):
                return imgs_trn.get(str(idx))[...]
            else:
                return imgs_tst.get(str(idx - len(imgs_trn)))[...]

        while True:

            rng.shuffle(didxs)
            didxs_cycle = cycle(didxs)

            for _ in range(nb_steps):
                ib = np.zeros(ib_shape, dtype=np.float16)

                for bidx in range(self.cfg['trn_batch_size']):
                    didx = next(didxs_cycle)
                    img = get_img(didx)
                    ib[bidx] = crop(img)

                ib = self.cfg['preprocess_imgs_func'](ib)
                yield ib, ib

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(AutoKNN())
