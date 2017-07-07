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


def _net_densenet121_sigmoid(input_shape=(224, 224, 3)):

    from planet.models.densenet.DN121 import densenet121_model
    from planet.utils.multi_gpu import make_parallel

    def preprocess_jpg(x):
        return x * 1. / 255.

    def preprocess_tags(x):
        return x

    dns = densenet121_model(img_rows=input_shape[0], img_cols=input_shape[1],
                            color_type=input_shape[2], dropout_rate=0.2)
    shared_out = dns.output
    x = Dense(len(TAGS))(shared_out)
    x = Activation('sigmoid')(x)
    model = Model(inputs=dns.input, outputs=x)

    return model, preprocess_jpg, preprocess_tags


def _net_densenet121_softmax(input_shape=(224, 224, 3)):

    from planet.models.densenet.DN121 import densenet121_model
    from planet.utils.multi_gpu import make_parallel

    def preprocess_jpg(x):
        return x * 1. / 255.

    def preprocess_tags(x):
        return x

    dns = densenet121_model(img_rows=input_shape[0], img_cols=input_shape[1], color_type=input_shape[2])
    shared_out = dns.output
    classifiers = [Dense(2, activation='softmax')(shared_out) for t in TAGS]
    x = concatenate(classifiers, axis=-1)
    x = Reshape((len(TAGS), 2))(x)
    x = Lambda(lambda x: x[:, :, 0])(x)
    model = Model(inputs=dns.input, outputs=x)

    return model, preprocess_jpg, preprocess_tags


class DenseNet121(object):

    def __init__(self):

        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/DenseNet121_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (140, 140, 3),
            'pp_imgs_func': lambda x: x,
            'pp_tags_func': lambda x: x,
            'imgs_resize_crop': 'resize',

            # Network setup.
            # 'net_builder_func': _net_densenet121_softmax,
            'net_builder_func': _net_densenet121_sigmoid,
            'net_loss_func': _loss_wbc,

            # Training setup.
            'trn_epochs': 100,
            'trn_augment_max_trn': 5,
            'trn_augment_max_val': 0,
            'trn_batch_size': 44,
            # 'trn_optimizer': Adam,
            # 'trn_optimizer_args': {'lr': 0.1},
            'trn_optimizer': SGD,
            'trn_optimizer_args': {'lr': 0.1, 'decay': 1e-4, 'momentum': 0.9, 'nesterov': True},
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

        # Network setup.
        net, self.cfg['pp_imgs_func'], self.cfg['pp_tags_func'] = \
            self.cfg['net_builder_func'](self.cfg['input_shape'])
        json.dump(net.to_json(), open('%s/model.json' % self.cpdir, 'w'), indent=2)
        json.dump(serialize_config(self.cfg), open('%s/config.json' % self.cpdir, 'w'))

        # Data setup.
        idxs_trn, idxs_val = get_train_val_idxs(self.cfg['hdf5_path_trn'], self.cfg['trn_prop_data'],
                                                self.cfg['trn_prop_trn'])
        steps_trn = max(math.ceil(len(idxs_trn) / self.cfg['trn_batch_size']), 400)
        steps_val = max(math.ceil(len(idxs_val) / self.cfg['trn_batch_size']), 20)
        gen_trn = self.batch_gen(idxs_trn, steps_trn, nb_augment_max=self.cfg['trn_augment_max_trn'])
        gen_val = self.batch_gen(idxs_val, steps_val, nb_augment_max=self.cfg['trn_augment_max_val'])

        opt = self.cfg['trn_optimizer'](**self.cfg['trn_optimizer_args'])
        net.compile(optimizer=opt, loss=self.cfg['net_loss_func'], metrics=[F2, prec, reca] + tag_metrics())

        net.summary()
        if weights_path is not None:
            net.load_weights(weights_path)
        pprint(self.cfg)

        cb = [
            ValidationCB(self.cfg['cpdir'], gen_val, self.cfg['trn_batch_size'], steps_val),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            EarlyStopping(monitor='F2', min_delta=0.01, patience=20, verbose=1, mode='max'),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalf2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='F2', factor=0.1, patience=5,
                              min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max')

        ] + callbacks

        if self.cfg['trn_monitor_val']:
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.01, patience=20, verbose=1, mode='max'))

        train = net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.cfg[
                                  'trn_epochs'], verbose=1, callbacks=cb)

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

        def crop(img):
            h, w, _ = self.cfg['input_shape']
            y0 = rng.randint(0, img.shape[0] - h)
            x0 = rng.randint(0, img.shape[1] - w)
            y1 = y0 + h
            x1 = x0 + w
            return img[y0:y1, x0:x1, :]

        while True:

            rng.shuffle(didxs)
            didxs_cycle = cycle(didxs)

            for _ in range(nb_steps):

                ib = np.zeros(ib_shape, dtype=np.float16)
                tb = np.zeros(tb_shape, dtype=np.int16)

                for bidx in range(self.cfg['trn_batch_size']):
                    didx = next(didxs_cycle)
                    img = imgs.get(str(didx))[...]
                    if self.cfg['imgs_resize_crop'] == 'resize':
                        ib[bidx] = imresize(img, self.cfg['input_shape'])
                    else:
                        ib[bidx] = crop(imgs.get(str(didx))[...])
                    tb[bidx] = tags[didx]
                    for aug in rng.choice(aug_funcs, rng.randint(0, nb_augment_max + 1)):
                        ib[bidx] = aug(ib[bidx])

                yield self.cfg['pp_imgs_func'](ib), self.cfg['pp_tags_func'](tb)

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(DenseNet121())
