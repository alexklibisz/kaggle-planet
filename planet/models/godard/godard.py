from itertools import cycle
from keras.optimizers import SGD, Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation, LocallyConnected1D, average
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN, Callback, LearningRateScheduler
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from pprint import pprint
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
from time import sleep, time
from tqdm import tqdm
import json
import h5py
import keras.backend as K
import logging
import math
import numpy as np
import os

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, correct_tags, get_train_val_idxs, serialize_config, IMG_MEAN_JPG_TRN, IMG_MEAN_TIF_TRN
from planet.utils.keras_utils import ValidationCB, HistoryPlotCB, tag_metrics, F2, prec, reca
from planet.utils.runtime import funcname
rng = np.random


def _loss_wbc(yt, yp):

    # Standard log loss.
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))

    # Compute weight matrix, scaled by the error at each tag.
    # Assumes a positive/negative threshold at 0.5.
    yp = K.round(yp)
    fnwgt, fpwgt = 3., 1.
    fnmat = K.clip(yt - yp, 0, 1) * (fnwgt - 1)
    fpmat = K.clip(yp - yt, 0, 1) * (fpwgt - 1)
    wmat = fnmat + fpmat + 1
    return K.mean(loss * wmat, axis=-1)

def _godard(inputs, num_outputs=len(TAGS), input_shape=(96,96,3), output_activation='sigmoid'):
    x = Lambda(lambda x: x * 1. / 255., input_shape=input_shape)(inputs)
    x = BatchNormalization(input_shape=input_shape)(x)

    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return Dense(num_outputs, activation=output_activation)(x)

def build_godard_sigmoid(input_shape=(96,96,3)):
    inputs = Input(shape=input_shape)
    outputs = _godard(inputs, input_shape=input_shape)
    return Model(inputs=inputs, outputs=outputs)

# This is a first pass at a network that combines outputs from other networks
def build_godard_ensemble(input_shape=(96,96,3), ensemble_size=2):
    inputs = Input(shape=input_shape)
    ensemble_outputs = [_godard(inputs, input_shape=input_shape) for i in range(ensemble_size)]
    for i,x in enumerate(ensemble_outputs):
        x = Reshape((17,1))(x)
        x = LocallyConnected1D(1, 1)(x)
        x = Reshape((17,))(x)
        ensemble_outputs[i] = x

    outputs = average(ensemble_outputs)
    return Model(inputs=inputs, outputs=outputs)


# This is an attempt to do what this paper said was possible: http://proceedings.mlr.press/v54/eban17a/eban17a.pdf
def f2_hinge_loss(yt, yp):
    allp = K.sum(yt)
    yp =  K.maximum(0.0, 1 - (yt*2-1)*(yp - 0.5)) # hinge loss
    tp = K.sum(yt * (1-yp))
    fp = K.sum((1-yt)*yp)
    b = 2
    return -(1 + b**2)*tp/(allp*(b**2) + tp + fp + K.epsilon())

def f2_loss(yt, yp):
    allp = K.sum(yt)
    tp = K.sum(yt * yp)
    fp = K.sum(K.clip(yp - yt, 0, 1))
    b = 2
    return -(1 + b**2)*tp/(allp*(b**2) + tp + fp + K.epsilon())

class Godard(object):

    def __init__(self, model_path=None):
        # Configuration.
        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/Godard_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (96, 96, 3),

            # Network setup.
            'net_builder_func': build_godard_sigmoid,
            'net_loss_func': 'binary_crossentropy',

            # Training setup.
            'trn_epochs': 30,
            'trn_augment_max_trn': 4,
            'trn_augment_max_val': 0,
            'trn_batch_size': 128,
            'trn_optimizer': Adam,
            'trn_optimizer_args': {'lr': 0.001},
            'trn_prop_trn': 0.8,
            'trn_prop_data': 1.0,
            'trn_monitor_val': False,

            # Testing.
            'tst_batch_size': 128
        }

        if model_path:
            self.net = load_model(model_path, compile=False)
            self.cfg['input_shape'] = self.net.input_shape[1:]
        else:
            self.net = self.cfg['net_builder_func'](self.cfg['input_shape'])
        #self.net.summary()

    @property
    def cpdir(self):
        if not os.path.exists(self.cfg['cpdir']):
            os.mkdir(self.cfg['cpdir'])
        return self.cfg['cpdir']

    def serialize(self):
        self.net.save('%s/model.hdf5' % self.cpdir)
        json.dump(serialize_config(self.cfg), open('%s/config.json' % self.cpdir, 'w'))
        pprint(self.cfg)

    def train(self, callbacks=[]):

        # Data setup.
        data = h5py.File(self.cfg['hdf5_path_trn'])
        tags = data.get('tags')[...]
        idxs_trn, idxs_val = get_train_val_idxs(tags, self.cfg['trn_prop_data'], self.cfg['trn_prop_trn'])
        steps_trn = math.ceil(len(idxs_trn) / self.cfg['trn_batch_size'] / self.cfg['trn_prop_data'])
        steps_val = math.ceil(len(idxs_val) / self.cfg['trn_batch_size'])
        gen_trn = self.batch_gen(data, idxs_trn, steps_trn, nb_augment_max=self.cfg['trn_augment_max_trn'])
        gen_val = self.batch_gen(data, idxs_val, steps_val, nb_augment_max=self.cfg['trn_augment_max_val'])

        opt = self.cfg['trn_optimizer'](**self.cfg['trn_optimizer_args'])
        self.net.compile(optimizer=opt, loss=self.cfg['net_loss_func'], metrics=[F2, prec, reca])
        # self.net.compile(optimizer=opt, loss=self.cfg['net_loss_func'], metrics=[F2, prec, reca] + tag_metrics())

        def lrsched(epoch):
            if epoch < 15:
                return 1e-3
            elif epoch < 23:
                return 1e-4
            else:
                return 1e-5

        cb = [
            ValidationCB(self.cfg['cpdir'], gen_val, self.cfg['trn_batch_size'], steps_val),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            # EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
            # EarlyStopping(monitor='F2', min_delta=0.01, patience=20, verbose=1, mode='max'),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/mvalf2_{epoch:02d}_{val_F2:.3f}.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            LearningRateScheduler(lrsched)
            # ReduceLROnPlateau(monitor='F2', factor=0.1, patience=5,
            #                   min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max')

        ] + callbacks

        if self.cfg['trn_monitor_val']:
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.01, patience=20, verbose=1, mode='max'))

        train = self.net.fit_generator(gen_trn, steps_per_epoch=steps_trn,
                                       epochs=self.cfg['trn_epochs'], verbose=1, callbacks=cb)

        return train.history

    def batch_gen(self, data, didxs, nb_steps, nb_augment_max, yield_didxs=False):

        imgs = data.get('images')
        tags = data.get('tags')[...]
        ib_shape = (self.cfg['trn_batch_size'],) + self.cfg['input_shape']
        tb_shape = (self.cfg['trn_batch_size'], len(TAGS))
        aug_funcs = [
            lambda x: x, lambda x: np.flipud(x), lambda x: np.fliplr(x),
            lambda x: np.rot90(x, rng.randint(1, 4)),
            # lambda x: np.roll(x, rng.randint(1, x.shape[0]), axis=rng.choice([0, 1]))
        ]

        while True:

            rng.shuffle(didxs)
            didxs_cycle = cycle(didxs)

            for _ in range(nb_steps):

                ib = np.zeros(ib_shape, dtype=np.float16)
                tb = np.zeros(tb_shape, dtype=np.int16)

                for bidx in range(self.cfg['trn_batch_size']):
                    didx = next(didxs_cycle)
                    ib[bidx] = imresize(imgs[didx, ...], self.cfg['input_shape'])
                    tb[bidx] = tags[didx]
                    for aug in rng.choice(aug_funcs, rng.randint(0, nb_augment_max + 1)):
                        ib[bidx] = aug(ib[bidx])

                yield ib, tb

    def predict_batch(self, imgs_batch, augment=False):
        """Predict a single batch of images, optionally with augmentation. Augmentations
        vectorized across the entire batch and predictions averaged."""
        if augment:
            aug_funcs = [
                lambda x: x,                                          # identity
                lambda x: x[:, ::-1, ...],                            # vlip
                lambda x: x[:, :, ::-1],                              # hflip
                lambda x: np.rot90(x, 1, axes=(1, 2)),                # +90
                lambda x: np.rot90(x, 2, axes=(1, 2)),                # +180
                lambda x: np.rot90(x, 3, axes=(1, 2)),                # +270
                lambda x: np.rot90(x, 1, axes=(1, 2))[:, ::-1, ...],  # vflip(+90)
                lambda x: np.rot90(x, 1, axes=(1, 2))[:, :, ::-1]     # vflip(+90)
            ]

            yp = np.zeros((imgs_batch.shape[0], len(TAGS)))
            for aug_func in aug_funcs:
                imgs_batch = aug_func(imgs_batch)
                tags_batch = self.net.predict(imgs_batch)
                yp += tags_batch / len(aug_funcs)
            return yp
        else:
            return self.net.predict_on_batch(imgs_batch)

    def predict(self, dataset):

        path = self.cfg['hdf5_path_trn'] if dataset == 'train' else self.cfg['hdf5_path_tst']
        data = h5py.File(path)
        yt = data.get('tags')[...]
        yp = np.zeros(yt.shape, dtype=np.float32)
        imgs = data.get('images')
        names = data.attrs['names'].split(',')
        imgs_batch = np.zeros((self.cfg['tst_batch_size'], *self.cfg['input_shape']), dtype=np.float32)

        for didx in tqdm(range(0, len(imgs), self.cfg['tst_batch_size'])):
            bsz = min(self.cfg['tst_batch_size'], len(imgs) - didx)
            for bidx in range(bsz):
                imgs_batch[bidx] = imresize(imgs[didx + bidx, ...], self.cfg['input_shape'])
            yp[didx:didx + bsz] = self.predict_batch(imgs_batch[:bsz])

        return names, yt, yp


if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(Godard)
