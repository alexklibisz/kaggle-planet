# Godard v2 adapted from LuckyLoser model.
from itertools import cycle
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN, Callback
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from pprint import pprint
from scipy.misc import imresize
from time import sleep, time
import json
import h5py
import keras.backend as K
import math
import numpy as np
import os

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, correct_tags, get_train_val_idxs, serialize_config
from planet.utils.keras_utils import TensorBoardWrapper
from planet.utils.runtime import funcname
rng = np.random


def _loss_bc(yt, yp):
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))
    return K.mean(loss, axis=-1)


def _net_godard(input_shape=(96, 96, 3)):

    from planet.utils.data_utils import IMG_MEAN_JPG_TRN, IMG_MEAN_TIF_TRN

    def preprocess_jpg(x):
        for c in range(x.shape[-1]):
            x[:, :, :, c] -= IMG_MEAN_JPG_TRN[c]
        return x / 255.

    def preprocess_tif(x):
        for c in range(x.shape[-1]):
            x[:, :, :, c] -= IMG_MEAN_TIF_TRN[c]
        return x / 65535.

    def preprocess_tags(x):
        return x

    input = Input(shape=input_shape, name='00.inpt')
    x = BatchNormalization(momentum=0.1, name='01.bnrm')(input)

    def conv_block(x, f, d, n='00'):
        x = Conv2D(f, 3, padding='same', kernel_initializer='he_normal', name='%s.0.conv.%d' % (n, f))(x)
        x = BatchNormalization(momentum=0.1, name='%s.1.bnrm' % n)(x)
        x = PReLU(name='%s.2.prelu' % n)(x)
        x = Conv2D(f, 3, padding='same', kernel_initializer='he_normal', name='%s.3.conv.%d' % (n, f))(x)
        x = BatchNormalization(momentum=0.1, name='%s.4.bnrm' % n)(x)
        x = PReLU(name='%s.5.prelu' % n)(x)
        x = MaxPooling2D(pool_size=2, name='%s.6.pool' % n)(x)
        x = Dropout(d, name='%s.7.drop' % n)(x)
        return x

    x = conv_block(x, 100, 0.2, '02.0')
    x = conv_block(x, 150, 0.2, '02.1')
    x = conv_block(x, 200, 0.2, '02.2')
    x = Flatten(name='02.5.flat')(x)

    # Shared dense layer.
    x = Dense(512, kernel_initializer='he_normal', name='03.dens.512')(x)
    x = BatchNormalization(momentum=0.1, name='04.bnrm')(x)
    x = PReLU(name='05.prelu')(x)
    shared = Dropout(0.2, name='06.drop')(x)

    # One classifier for the four cloud-cover tags.
    x = Dense(4, kernel_initializer='glorot_uniform', name='07.clouds.dens')(shared)
    x = BatchNormalization(momentum=0.1, name='07.clouds.bnrm')(x)
    out_clouds = Activation('softmax', name='07.clouds.soft')(x)

    # Re-ordered arrangement of tags.
    net_tags = ['clear', 'cloudy', 'haze', 'partly_cloudy', 'agriculture', 'artisinal_mine',
                'bare_ground', 'blooming', 'blow_down', 'conventional_mine', 'cultivation',
                'habitation', 'primary', 'road', 'selective_logging', 'slash_burn', 'water']

    # One softmax classifier for each of the remaining thirteen tags.
    out_others = []
    for tag in net_tags[4:]:
        x = Dense(2, kernel_initializer='glorot_uniform', name='08.%s.dens' % tag)(shared)
        x = BatchNormalization(momentum=0.1, name='08.%s.bnrm' % tag)(x)
        x = Activation('softmax', name='08.%s.soft' % tag)(x)
        x = Lambda(lambda x: K.expand_dims(x[:, 1]), name='09.%s' % tag)(x)
        out_others.append(x)

    # Combine the activations.
    combined = []
    for i, tag in enumerate(net_tags[:4]):
        x = Lambda(lambda x: K.expand_dims(x[:, i]), name='09.%s' % tag)(out_clouds)
        combined.append(x)
    combined += out_others

    # Re-arrange and concatenate them to match the original order.
    arranged = concatenate([combined[net_tags.index(t)] for t in TAGS], name='10.concat')

    return Model(inputs=input, outputs=arranged), preprocess_jpg if input_shape[-1] == 3 else preprocess_tif, preprocess_tags


class GodardV2(object):

    def __init__(self):

        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/godardv2_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (100, 100, 3),
            'preprocess_imgs_func': lambda x: x,
            'preprocess_tags_func': lambda x: x,

            # Network setup.
            'net_builder_func': _net_godard,
            'net_loss_func': _loss_bc,
            'net_threshold': 0.5,

            # Training setup.
            'trn_epochs': 100,
            'trn_augment_max_trn': 5,
            'trn_augment_max_val': 1,
            'trn_batch_size': 40,
            'trn_adam_params': {'lr': 0.0015},
            'trn_prop_trn': 0.8,
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
        net, self.cfg['preprocess_imgs_func'], self.cfg['preprocess_tags_func'] = \
            self.cfg['net_builder_func'](self.cfg['input_shape'])
        json.dump(net.to_json(), open('%s/model.json' % self.cpdir, 'w'), indent=2)
        json.dump(serialize_config(self.cfg), open('%s/config.json' % self.cpdir, 'w'))

        # Data setup.
        idxs_trn, idxs_val = get_train_val_idxs(self.cfg['hdf5_path_trn'], prop_data=self.cfg['trn_prop_data'],
                                                prop_trn=self.cfg['trn_prop_trn'])
        steps_trn = math.ceil(len(idxs_trn) / self.cfg['trn_batch_size'])
        steps_val = math.ceil(len(idxs_val) / self.cfg['trn_batch_size'])
        gen_trn = self.batch_gen(idxs_trn, steps_trn, nb_augment_max=self.cfg['trn_augment_max_trn'])
        gen_val = self.batch_gen(idxs_val, steps_val, nb_augment_max=self.cfg['trn_augment_max_val'])

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

        def yppos(yt, yp):
            ypr = K.cast(yp > self.cfg['net_threshold'], 'float')
            return K.sum(ypr * yp) / K.sum(K.round(ypr))

        def ypneg(yt, yp):
            ypr = K.cast(yp > self.cfg['net_threshold'], 'float')
            yprneg = K.abs(ypr - 1)
            return K.sum(yprneg * yp) / K.sum(yprneg)

        def ytmean(yt, yp):
            return K.mean(yt)

        def ypmean(yt, yp):
            yp = K.cast(yp > self.cfg['net_threshold'], 'float')
            return K.mean(yp)

        net.compile(optimizer=Adam(**self.cfg['trn_adam_params']), loss=self.cfg['net_loss_func'],
                    metrics=[F2, prec, reca, yppos, ypneg, ytmean, ypmean])

        net.summary()
        if weights_path is not None:
            net.load_weights(weights_path)
        pprint(self.cfg)

        cb = [
            EarlyStopping(monitor='F2', min_delta=0.01, patience=20, verbose=1, mode='max'),
            TensorBoardWrapper(gen_val, nb_steps=2, log_dir=self.cfg['cpdir'], histogram_freq=1,
                               batch_size=4, write_graph=False, write_grads=True),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalf2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            TerminateOnNaN()

        ] + callbacks

        if self.cfg['trn_monitor_val']:
            cb.append(ReduceLROnPlateau(monitor='val_F2', factor=0.1, patience=10,
                                        min_lr=0.00001, epsilon=1e-2, verbose=1, mode='max'))
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.01, patience=20, verbose=1, mode='max'))

        train = net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.cfg['trn_epochs'],
                                  verbose=1, callbacks=cb, validation_data=gen_val, validation_steps=steps_val)

        return train.history

    def batch_gen(self, didxs, nb_steps, nb_augment_max, yield_didxs=False):

        data = h5py.File(self.cfg['hdf5_path_trn'], 'r')
        imgs = data.get('images')
        tags = data.get('tags')[:, :]
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
    model_runner(GodardV2())
