# Godard v2 adapted from LuckyLoser model.
from itertools import cycle
from keras.optimizers import SGD, Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN, Callback
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from pprint import pprint
from scipy.misc import imresize
from scipy.ndimage.interpolation import rotate
from time import sleep, time
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
from planet.utils.keras_utils import TensorBoardWrapper, HistoryPlotCB
from planet.utils.runtime import funcname
rng = np.random


class FineTuneCB(Callback):

    def __init__(self, unfreeze_after=5, unfreeze_lr_mult=0.5):
        super().__init__()
        self.unfreeze_after = unfreeze_after
        self.unfreeze_lr_mult = unfreeze_lr_mult

    def on_epoch_begin(self, epoch, logs):
        if epoch == self.unfreeze_after:
            logger = logging.getLogger(funcname())
            logger.info('Epoch %d: unfreezing layers.' % epoch)
            for layer in self.model.layers:
                layer.trainable = True
            lr = K.get_value(self.model.optimizer.lr) * self.unfreeze_lr_mult
            K.set_value(self.model.optimizer.lr, lr)
            logger.info('Epoch %d: new learning rate %.4lf.' % (epoch, K.get_value(self.model.optimizer.lr)))


def _loss_bc(yt, yp):

    # Standard log loss.
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))
    # return K.mean(loss, axis=-1)

    # Compute weight matrix, scaled by the error at each tag.
    fnmax = 20.
    fnmat = K.clip(yt - yp, 0, 1) * fnmax
    fpmat = K.abs(yt - 1)
    wmat = fnmat + fpmat
    return K.mean(loss * wmat, axis=-1)


def _net_resnet50(input_shape=(200, 200, 3)):

    from keras.applications.resnet50 import ResNet50

    def preprocess_jpg(x):
        for c in range(x.shape[-1]):
            x[:, :, :, c] -= IMG_MEAN_JPG_TRN[c]
        return x / 255.

    def preprocess_tags(x):
        return x

    res = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')
    x = Flatten()(res.output)
    shared_out = Dropout(0.1)(x)

    # Add a classifier for each class instead of a single classifier.
    # Concatenate classifiers and reshape to match output shape.
    classifiers = [Dense(2, activation='softmax')(shared_out) for t in TAGS]
    x = concatenate(classifiers, axis=-1)
    x = Reshape((17, 2))(x)
    x = Lambda(lambda x: x[:, :, 1])(x)

    model = Model(inputs=res.input, outputs=x)

    # Initially freeze the resnet layers.
    for l in model.layers[:len(res.layers)]:
        l.trainable = False

    return model, preprocess_jpg, preprocess_tags


class ResNet50(object):

    def __init__(self):

        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/ResNet50_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (224, 224, 3),
            'preprocess_imgs_func': lambda x: x,
            'preprocess_tags_func': lambda x: x,

            # Network setup.
            'net_builder_func': _net_resnet50,
            'net_loss_func': _loss_bc,
            'net_threshold': 0.5,

            # Training setup.
            'trn_epochs': 100,
            'trn_augment_max_trn': 5,
            'trn_augment_max_val': 1,
            'trn_batch_size': 32,
            'trn_optimizer': Adam,
            'trn_optimizer_args': {'lr': 0.01},
            'trn_prop_trn': 0.9,
            'trn_prop_data': 1.0,
            'trn_monitor_val': True,

            # Testing.
            'tst_batch_size': 1000
        }

    @property
    def cpdir(self):
        if not os.path.exists(self.cfg['cpdir']):
            os.mkdir(self.cfg['cpdir'])
        return self.cfg['cpdir']

    def train(self, weights_path=None, callbacks=[]):

        # Metrics.
        def prec(yt, yp):
            yp = K.cast(yp > self.cfg['net_threshold'], 'float')
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            return tp / (tp + fp + K.epsilon())

        def reca(yt, yp):
            yp = K.cast(yp > self.cfg['net_threshold'], 'float')
            tp = K.sum(yt * yp)
            fn = K.sum(K.clip(yt - yp, 0, 1))
            return tp / (tp + fn + K.epsilon())

        def F2(yt, yp):
            p = prec(yt, yp)
            r = reca(yt, yp)
            b = 2.0
            return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))

        # Network setup. ResNet layers frozen at first.
        net, self.cfg['preprocess_imgs_func'], self.cfg['preprocess_tags_func'] = \
            self.cfg['net_builder_func'](self.cfg['input_shape'])
        json.dump(net.to_json(), open('%s/model.json' % self.cpdir, 'w'), indent=2)
        json.dump(serialize_config(self.cfg), open('%s/config.json' % self.cpdir, 'w'))

        # Data setup.
        idxs_trn, idxs_val = get_train_val_idxs(self.cfg['hdf5_path_trn'], self.cfg['trn_prop_data'],
                                                self.cfg['trn_prop_trn'])
        steps_trn = math.ceil(len(idxs_trn) / self.cfg['trn_batch_size'])
        steps_val = math.ceil(len(idxs_val) / self.cfg['trn_batch_size'])
        gen_trn = self.batch_gen(idxs_trn, steps_trn, nb_augment_max=self.cfg['trn_augment_max_trn'])
        gen_val = self.batch_gen(idxs_val, steps_val, nb_augment_max=self.cfg['trn_augment_max_val'])

        opt = self.cfg['trn_optimizer'](**self.cfg['trn_optimizer_args'])
        net.compile(optimizer=opt, loss=self.cfg['net_loss_func'], metrics=[F2, prec, reca])

        net.summary()
        if weights_path is not None:
            net.load_weights(weights_path)
        pprint(self.cfg)

        cb = [
            FineTuneCB(unfreeze_after=5, unfreeze_lr_mult=0.1),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            EarlyStopping(monitor='F2', min_delta=0.01, patience=20, verbose=1, mode='max'),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalf2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            TerminateOnNaN(),

        ] + callbacks

        if self.cfg['trn_monitor_val']:
            cb.append(ReduceLROnPlateau(monitor='val_F2', factor=0.5, patience=2,
                                        min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max'))
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.01, patience=20, verbose=1, mode='max'))

        train = net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.cfg['trn_epochs'], verbose=1, callbacks=cb,
                                  validation_data=gen_val, validation_steps=steps_val, workers=1)

        return train.history

    def batch_gen(self, didxs, nb_steps, nb_augment_max, yield_didxs=False):

        data = h5py.File(self.cfg['hdf5_path_trn'], 'r')
        imgs = data.get('images')
        tags = data.get('tags')[:, :]
        ib_shape = (self.cfg['trn_batch_size'],) + self.cfg['input_shape']
        tb_shape = (self.cfg['trn_batch_size'], len(TAGS))
        aug_funcs = [
            lambda x: x, lambda x: np.flipud(x), lambda x: np.fliplr(x),
            lambda x: np.rot90(x, rng.randint(1, 4)),
            lambda x: np.roll(x, rng.randint(1, x.shape[0]), axis=rng.choice([0, 1]))
        ]

        while True:

            rng.shuffle(didxs)
            didxs_cycle = cycle(didxs)

            for _ in range(nb_steps):

                ib = np.zeros(ib_shape, dtype=np.float16)
                tb = np.zeros(tb_shape, dtype=np.int16)
                db = np.zeros((self.cfg['trn_batch_size'],), dtype=np.uint16)

                for bidx in range(self.cfg['trn_batch_size']):
                    db[bidx] = next(didxs_cycle)
                    img = imgs.get(str(db[bidx]))[...]
                    ib[bidx] = imresize(img, self.cfg['input_shape'])
                    tb[bidx] = tags[db[bidx]]
                    for aug in rng.choice(aug_funcs, rng.randint(0, nb_augment_max + 1)):
                        ib[bidx] = aug(ib[bidx])

                yield self.cfg['preprocess_imgs_func'](ib), self.cfg['preprocess_tags_func'](tb)

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(ResNet50())
