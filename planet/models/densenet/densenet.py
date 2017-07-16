from itertools import cycle
from keras.optimizers import SGD, Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation
from keras.models import Model, model_from_json, load_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN, Callback
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from pprint import pprint
from psutil import virtual_memory
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
import pandas as pd

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, correct_tags, get_train_val_idxs, serialize_config, IMG_MEAN_JPG_TRN, IMG_STDV_JPG_TRN
from planet.utils.keras_utils import ValidationCB, HistoryPlotCB, tag_metrics, F2, prec, reca
from planet.utils.runtime import funcname
from planet.utils.multi_gpu import make_parallel
rng = np.random


def _loss_wbc(yt, yp):

    # Standard log loss. One scaler for every sample,tag.
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))

    # Probability of each tag occurring across full dataset.
    prob = [0.304, 0.008, 0.021, 0.008, 0.002, 0.702, 0.052, 0.002,
            0.111, 0.090, 0.067, 0.179, 0.927, 0.199, 0.008, 0.005, 0.183]

    # Probability-based weights higher for lower-probability tags.
    wprob = 2. - np.asarray(prob)

    # Baseline weight for ground-truth positives.
    wposi = 2.

    # Apply the common baseline weight and scale by the probability weights.
    # Add yt inverse to account for ground-truth negatives.
    wmat = yt * wposi * wprob + (1 - yt)

    # Apply weights.
    return K.mean(loss * wmat, axis=-1)


class FineTuneCB(Callback):

    def __init__(self, frozen_up_to_idx=0, unfreeze_epoch=-1, unfreeze_lr_mult=1):
        super().__init__()
        self.frozen_up_to_idx = frozen_up_to_idx
        self.unfreeze_epoch = unfreeze_epoch
        self.unfreeze_lr_mult = unfreeze_lr_mult

    def on_train_begin(self, logs):
        logger = logging.getLogger(funcname())
        for idx, layer in enumerate(self.model.layers[:self.frozen_up_to_idx]):
            layer.trainable = False
            logger.info('Freezing layer %d: %s' % (idx, layer.name))

    def on_epoch_begin(self, epoch, logs):
        if epoch == self.unfreeze_epoch:
            logger = logging.getLogger(funcname())
            for idx, layer in enumerate(self.model.layers):
                layer.trainable = True
                logger.info('Unfreezing layer %d: %s' % (idx, layer.name))
            lr = K.get_value(self.model.optimizer.lr) * self.unfreeze_lr_mult
            K.set_value(self.model.optimizer.lr, lr)
            logger.info('Epoch %d: new learning rate %.4lf.' % (epoch, K.get_value(self.model.optimizer.lr)))


def preprocess_pretrained(x):
    """Switch RGB -> BGR and follow preprocessing from: https://goo.gl/nR2j6D and https://goo.gl/a8fkve"""
    B = (x[:, :, :, 2:3] * 1. - 103.94) * 0.017
    G = (x[:, :, :, 1:2] * 1. - 116.78) * 0.017
    R = (x[:, :, :, 0:1] * 1. - 123.68) * 0.017
    return K.concatenate([B, G, R], axis=-1)


def preprocess(x):
    """Mean subtraction, normalization based on training dataset."""
    R = (x[:, :, :, 0:1] * 1. - IMG_MEAN_JPG_TRN[0]) / IMG_STDV_JPG_TRN[0]
    G = (x[:, :, :, 1:2] * 1. - IMG_MEAN_JPG_TRN[1]) / IMG_STDV_JPG_TRN[1]
    B = (x[:, :, :, 2:3] * 1. - IMG_MEAN_JPG_TRN[2]) / IMG_STDV_JPG_TRN[2]
    return K.concatenate([B, G, R], axis=-1)


def _densenet121(input_shape=(256, 256, 3), pretrained=False):

    from planet.models.densenet.DN121 import densenet121_model

    ppf = preprocess_pretrained if pretrained else preprocsss
    preprocess = Lambda(ppf, input_shape=input_shape, name='preprocsss')
    dns = densenet121_model(img_rows=input_shape[0], img_cols=input_shape[1], color_type=input_shape[2],
                            dropout_rate=0.0, pretrained=pretrained, preprocess_layer=preprocess)
    shared_out = dns.output
    x = Dense(len(TAGS), kernel_initializer='he_normal')(shared_out)
    x = Activation('sigmoid')(x)
    model = Model(inputs=dns.input, outputs=x)

    return model


def _densenet169(input_shape=(256, 256, 3), pretrained=False):

    from planet.models.densenet.DN169 import densenet169_model

    ppf = preprocess_pretrained if pretrained else preprocsss
    preprocess = Lambda(ppf, input_shape=input_shape, name='preprocsss')
    dns = densenet169_model(img_rows=input_shape[0], img_cols=input_shape[1], color_type=input_shape[2],
                            dropout_rate=0.2, pretrained=pretrained, preprocess_layer=preprocess)
    shared_out = dns.output
    x = Dense(len(TAGS), kernel_initializer='he_normal')(shared_out)
    x = Activation('sigmoid')(x)
    model = Model(inputs=dns.input, outputs=x)
    return model


def _densenet121_pretrained(input_shape=(256, 256, 3)):
    return _densenet121(input_shape, pretrained=True)


def _densenet169_pretrained(input_shape=(256, 256, 3)):
    return _densenet169(input_shape, pretrained=True)


class DenseNet121(object):

    def __init__(self, model_path=None):

        # Configuration.
        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/DenseNet_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (256, 256, 3),

            # Network setup.
            # 'net_builder_func': _densenet169,
            # 'net_builder_func': _densenet121,
            # 'trn_optimizer': SGD,
            # 'trn_optimizer_args': {'lr': 0.1, 'decay': 1e-4, 'momentum': 0.9, 'nesterov': 1},
            # 'trn_optimizer': Adam,
            # 'trn_optimizer_args': {'lr': 0.002},

            # 'net_builder_func': _densenet169_pretrained,
            'net_builder_func': _densenet121_pretrained,
            'trn_optimizer': SGD,
            'trn_optimizer_args': {'lr': 0.001, 'decay': 1e-6, 'momentum': 0.9, 'nesterov': 1},
            'net_loss_func': _loss_wbc,
            'trn_finetune_args': {},
            # 'trn_finetune_args': {'frozen_up_to_idx': -2, 'unfreeze_epoch': 1, 'unfreeze_lr_mult': 1.},

            # Training setup.
            'trn_epochs': 100,
            'trn_augment_max_trn': 10,
            'trn_augment_max_val': 0,
            'trn_batch_size': 16,
            'trn_prop_trn': 0.9,
            'trn_prop_data': 1.0,
            'trn_monitor_val': True,

            # Testing.
            'tst_batch_size': 500,

            'seed': os.getpid()
        }

        if model_path:
            from planet.models.densenet.scale_layer import Scale
            self.net = load_model(model_path, {
                'Scale': Scale,
                'IMG_MEAN_JPG_TRN': IMG_MEAN_JPG_TRN,
                'IMG_STDV_JPG_TRN': IMG_STDV_JPG_TRN,
            }, compile=False)
            self.cfg['input_shape'] = self.net.input_shape[1:]
        else:
            self.net = self.cfg['net_builder_func'](self.cfg['input_shape'])

        # Model parallelism.
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        if len(gpus) > 1:
            self.net = make_parallel(self.net, len(gpus))
            self.cfg['trn_batch_size'] *= len(gpus)

        np.random.seed(self.cfg['seed'])

    @property
    def cpdir(self):
        if not os.path.exists(self.cfg['cpdir']):
            os.mkdir(self.cfg['cpdir'])
        return self.cfg['cpdir']

    def serialize(self):
        self.net.save('%s/model.hdf5' % self.cpdir)
        json.dump(serialize_config(self.cfg), open('%s/config.json' % self.cpdir, 'w'))

    def train(self, callbacks=[]):

        # Data setup.
        data = h5py.File(self.cfg['hdf5_path_trn'])
        tags = data.get('tags')[...]
        idxs_trn, idxs_val = get_train_val_idxs(tags, self.cfg['trn_prop_data'], self.cfg['trn_prop_trn'])
        steps_trn = math.ceil(len(idxs_trn) / self.cfg['trn_batch_size'])
        steps_val = math.ceil(len(idxs_val) / self.cfg['trn_batch_size'])
        gen_trn = self.batch_gen(data, idxs_trn, steps_trn, self.cfg['trn_augment_max_trn'])
        gen_val = self.batch_gen(data, idxs_val, steps_val, self.cfg['trn_augment_max_val'])

        opt = self.cfg['trn_optimizer'](**self.cfg['trn_optimizer_args'])
        self.net.compile(optimizer=opt, loss=self.cfg['net_loss_func'], metrics=[F2, prec, reca] + tag_metrics())
        pprint(self.cfg)

        cb = [
            FineTuneCB(**self.cfg['trn_finetune_args']),
            ValidationCB(self.cfg['cpdir'], gen_val, self.cfg['trn_batch_size'], steps_val),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            EarlyStopping(monitor='F2', min_delta=0.01, patience=20, verbose=1, mode='max'),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/model_{val_F2:.4f}_{epoch:02d}.hdf5' % self.cpdir,
                            monitor='val_F2', verbose=1, save_best_only=False, mode='max'),
            ReduceLROnPlateau(monitor='F2', factor=0.1, patience=5,
                              min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max')
        ] + callbacks

        if self.cfg['trn_monitor_val']:
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.01, patience=20, verbose=1, mode='max'))

        train = self.net.fit_generator(gen_trn, steps_per_epoch=steps_trn,
                                       epochs=self.cfg['trn_epochs'], verbose=1, callbacks=cb)

        return train.history

    def batch_gen(self, data, didxs, nb_steps, nb_augment_max):

        ram = virtual_memory().total / 1024**3
        imgs = data.get('images') if ram < 30 else data.get('images')[...]
        tags = data.get('tags')[...]
        ib_shape = (self.cfg['trn_batch_size'],) + self.cfg['input_shape']
        tb_shape = (self.cfg['trn_batch_size'], len(TAGS))
        aug_funcs = [
            lambda x: x, lambda x: np.flipud(x), lambda x: np.fliplr(x),
            lambda x: np.rot90(x, rng.randint(1, 4))
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

    def predict_batch(self, imgs_batch):
        return self.net.predict(imgs_batch)

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(DenseNet121)
