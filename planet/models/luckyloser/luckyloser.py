# Let's just try some random shit.
from itertools import cycle
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN, Callback
from keras.layers.advanced_activations import PReLU
from keras.losses import binary_crossentropy
from keras.regularizers import l2
from pprint import pprint
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize
from time import sleep, time
from tqdm import tqdm
import json
import h5py
import keras.backend as K
import math
import numpy as np
import os
import pickle as pkl

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, tagstr_to_binary, correct_tags, get_train_val_idxs
from planet.utils.keras_utils import HistoryPlotCB, ValidationCB, ThresholdedSigmoid, TensorBoardWrapper
from planet.utils.runtime import funcname
rng = np.random

# Composable functions for structuring network.


def serialize_config(config):
    _config = {}
    for k, v in config.items():
        if callable(v):
            _config[k] = v.__name__
        elif type(v) == np.int64:
            _config[k] = float(v)
        else:
            _config[k] = v
    return _config


def _out_single_sigmoid(input, prop_dropout=None):
    '''Apply a 17x1 sigmoid activation with batch normalization to the input.'''
    x = BatchNormalization()(input)
    x = Dropout(rng.uniform(0.05, 0.5) if prop_dropout is None else prop_dropout)(x)
    return Dense(17, activation='sigmoid')(x)


def _out_multi_softmax(input, prop_dropout=None):
    '''17 individual softmax classifiers. Spliced to only include the positive activation.'''
    x = BatchNormalization()(input)
    _x = Dropout(rng.uniform(0.05, 0.5) if prop_dropout is None else prop_dropout)(x)
    x = [Dense(2, activation='softmax')(_x) for _ in TAGS]
    x = concatenate(x)
    x = Reshape((len(TAGS), 2))(x)
    return Lambda(lambda x: x[:, :, 1])(x)


def _out_softmax_cover_sigmoid_rest():
    return

outputs = [_out_single_sigmoid, _out_multi_softmax]


def _loss_bcr(yt, yp):
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))
    return K.mean(loss, axis=-1)


def _loss_bcr_balanced(yt, yp):
    return


def _loss_bcr_wfn(yt, yp):
    return


def _loss_mse(yt, yp):
    return K.mean((yt - yp) ** 2, axis=-1)


def _loss_mae(yt, yp):
    return K.mean(K.abs(yt - yp), axis=-1)

losses = [_loss_bcr, _loss_mse]


def _net_vgg19(output=None, pretrained=False):

    from keras.applications.imagenet_utils import preprocess_input
    from keras.applications.vgg19 import VGG19

    vgg = VGG19(include_top=True, weights='imagenet' if pretrained else None)
    out = output(vgg.get_layer('fc2').output)
    net = Model(inputs=vgg.input, outputs=out)

    return net, net.input_shape[1:], preprocess_input


def _net_vgg19_pretrained(output=None):
    return _net_vgg19(output, pretrained=True)


def _net_vgg16(output=None, pretrained=False):

    from keras.applications.imagenet_utils import preprocess_input
    from keras.applications.vgg16 import VGG16

    vgg = VGG16(include_top=True, weights='imagenet' if pretrained else None)
    out = output(vgg.get_layer('fc2').output)
    net = Model(inputs=vgg.input, outputs=out)

    return net, net.input_shape[1:], preprocess_input


def _net_vgg16_pretrained(output=None):
    return _net_vgg16(output, pretrained=True)


def _net_godard(output_func=None, input_shape=(64, 64, 3)):

    from planet.utils.data_utils import IMG_MEAN_JPG_TRN, IMG_MEAN_TIF_TRN

    def mean_subtract_imgs(x):
        if x.shape[-1] == 3:
            for c in range(x.shape[-1]):
                x[:, :, :, c] -= IMG_MEAN_JPG_TRN[c]
            return x / 255.
        else:
            for c in range(x.shape[-1]):
                x[:, :, :, c] -= IMG_MEAN_TIF_TRN[c]
            return x / 65535.

    def tags(x):
        return x

    input = Input(shape=input_shape, name='00.inpt')
    x = BatchNormalization(momentum=0.1, name='01.bnrm')(input)

    def conv_block(x, f, d, n='00'):
        x = Conv2D(f, 3, padding='same', kernel_initializer='he_uniform', name='%s.0.conv.%d' % (n, f))(x)
        x = BatchNormalization(momentum=0.1, name='%s.1.bnrm' % n)(x)
        x = PReLU(name='%s.2.prelu' % n)(x)
        x = Conv2D(f, 3, padding='same', kernel_initializer='he_uniform', name='%s.3.conv.%d' % (n, f))(x)
        x = BatchNormalization(momentum=0.1, name='%s.4.bnrm' % n)(x)
        x = PReLU(name='%s.5.prelu' % n)(x)
        x = MaxPooling2D(pool_size=2, name='%s.6.pool' % n)(x)
        x = Dropout(d, name='%s.7.drop' % n)(x)
        return x

    x = conv_block(x, 32, 0.1, '02.0')
    x = conv_block(x, 64, 0.1, '02.1')
    x = conv_block(x, 128, 0.1, '02.2')
    x = conv_block(x, 256, 0.1, '02.3')
    x = Flatten(name='02.4.flat')(x)

    x = Dense(512, kernel_initializer='he_uniform', name='03.dens.512')(x)
    x = BatchNormalization(momentum=0.1, name='04.bnrm')(x)
    x = PReLU(name='05.prelu')(x)
    x = Dropout(0.1, name='06.drop')(x)

    x = Dense(len(TAGS), kernel_initializer='glorot_uniform', name='07.dense.17')(x)
    x = BatchNormalization(beta_regularizer=l2(0.0), momentum=0.1, name='08.bnrm')(x)
    x = Activation('sigmoid', name='09.sigm')(x)

    return Model(inputs=input, outputs=x), input_shape, mean_subtract_imgs, tags


nets = [_net_godard]


class LuckyLoser(object):

    def __init__(self):

            # One big configuration object for easy parameter searching.
        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/luckyloser_%d_%d' % (int(time()), os.getpid()),
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (96, 96, 3),
            'preprocess_imgs_func': None,
            'preprocess_tags_func': None,

            # Network setup.
            'net_builder_func': _net_godard,
            'net_out_func': _out_single_sigmoid,
            'net_loss_func': _loss_bcr,
            'net_threshold': 0.5,

            # Training setup.
            'trn_epochs': 100,
            'trn_augment_max_trn': 5,
            'trn_augment_max_val': 1,
            'trn_batch_size': 40,
            'trn_adam_params': {'lr': 0.001},
            'trn_prop_trn': 0.8,
            'trn_prop_data': 0.3,
            'trn_monitor_val': False,

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
        net, inshp, ppimgs, pptags = self.cfg['net_builder_func'](self.cfg['net_out_func'], self.cfg['input_shape'])
        self.cfg['input_shape'] = inshp
        self.cfg['preprocess_imgs_func'] = ppimgs
        self.cfg['preprocess_tags_func'] = pptags
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

        def acc(yt, yp):
            fp = K.sum(K.clip(yp - yt, 0, 1))
            fn = K.sum(K.clip(yt - yp, 0, 1))
            sz = K.sum(K.ones_like(yt))
            return (sz - (fp + fn)) / (sz + 1e-7)

        net.compile(optimizer=Adam(**self.cfg['trn_adam_params']), loss=self.cfg['net_loss_func'],
                    metrics=[F2, prec, reca, acc])

        net.summary()
        if weights_path is not None:
            net.load_weights(weights_path)
        pprint(self.cfg)

        cb = [
            EarlyStopping(monitor='F2', min_delta=0.01, patience=5, verbose=1, mode='max'),
            TensorBoardWrapper(gen_val, nb_steps=5, log_dir=self.cfg['cpdir'], histogram_freq=1,
                               batch_size=4, write_graph=False, write_grads=True),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalf2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            TerminateOnNaN()

        ] + callbacks

        if self.cfg['trn_monitor_val']:
            cb.append(ReduceLROnPlateau(monitor='val_F2', factor=0.75, patience=10,
                                        min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max'))
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.01, patience=10, verbose=1, mode='max'))

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
            # lambda x: rotate(x, rng.randint(0, 360), reshape=False, mode='reflect')
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
                    for aug in rng.choice(aug_funcs, rng.randint(0, nb_augment_max)):
                        ib[bidx] = aug(ib[bidx])

                yield self.cfg['preprocess_imgs_func'](ib), self.cfg['preprocess_tags_func'](tb)

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(LuckyLoser())
