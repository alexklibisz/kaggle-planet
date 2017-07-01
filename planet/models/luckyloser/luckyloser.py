# Let's just try some random shit.
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization, Activation
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, TerminateOnNaN
from keras.layers.advanced_activations import PReLU, ELU
from keras.losses import binary_crossentropy
from pprint import pprint
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize
from time import sleep
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
from planet.utils.data_utils import TAGS, IMG_MEAN_JPG_TRN, tagstr_to_binary, correct_tags, get_train_val_idxs
from planet.utils.keras_utils import HistoryPlotCB, ValidationCB, prec, reca, F2
from planet.utils.runtime import funcname
rng = np.random

# Composable functions for structuring network.


def _out_single_sigmoid(input, prop_dropout=None):
    '''Apply a 17x1 sigmoid activation with batch normalization to the input.'''
    x = BatchNormalization()(input)
    x = Dropout(rng.uniform(0.05, 0.5) if prop_dropout is None else prop_dropout)(x)
    return Dense(17, activation='sigmoid')(x)


def _out_multi_softmax():
    return


def _out_softmax_cover_sigmoid_rest():
    return


def _loss_bcr(yt, yp):
    loss = -1 * (yt * K.log(yp + 1e-7) + (1 - yt) * K.log(1 - yp + 1e-7))
    return K.mean(loss, axis=-1)


def _loss_bcr_balanced(yt, yp):
    return


def _loss_bcr_wfn(yt, yp):
    return


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


def _net_godard(output=None):
    return


class LuckyLoser(object):

    def __init__(self):

            # One big configuration object for easy parameter searching.
        self.cfg = {

            # Data setup.
            'cpdir': 'checkpoints/luckyloser',
            'hdf5_path_trn': 'data/train-jpg.hdf5',
            'hdf5_path_tst': 'data/test-jpg.hdf5',
            'input_shape': (150, 150, 3),
            'preprocess_func': None,

            # Network setup.
            'net_builder_func': _net_vgg16_pretrained,
            'net_out_func': _out_single_sigmoid,
            'net_loss_func': _loss_bcr,

            # Training setup.
            'trn_epochs': 50,
            'trn_augment_max_trn': 10,
            'trn_augment_max_val': 2,
            'trn_batch_size': 32,
            'trn_adam_params': {'lr': 0.002},
            'trn_prop_trn': 0.8,
            'trn_prop_data': 1.,
            'trn_early_stop': True,

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
        net, inshp, ppin = self.cfg['net_builder_func'](self.cfg['net_out_func'])
        self.cfg['input_shape'] = inshp
        self.cfg['preprocess_func'] = ppin
        json.dump(net.to_json(), open('%s/model.json' % self.cpdir, 'w'))

        # Data setup.
        idxs_trn, idxs_val = get_train_val_idxs(self.cfg['hdf5_path_trn'], prop_data=self.cfg['trn_prop_data'],
                                                prop_trn=self.cfg['trn_prop_trn'])
        steps_trn = math.ceil(len(idxs_trn) / self.cfg['trn_batch_size'])
        steps_val = math.ceil(len(idxs_val) / self.cfg['trn_batch_size'])
        gen_trn = self.batch_gen(idxs_trn, steps_trn, nb_augment_max=self.cfg['trn_augment_max_trn'])
        gen_val = self.batch_gen(idxs_val, steps_val, nb_augment_max=self.cfg['trn_augment_max_val'])

        net.compile(optimizer=Adam(**self.cfg['trn_adam_params']), loss=self.cfg['net_loss_func'],
                    metrics=[F2, prec, reca, 'accuracy'])

        net.summary()
        if weights_path is not None:
            net.load_weights(weights_path)
        pprint(self.cfg)

        cb = [
            ValidationCB(self.cpdir, gen_val, self.cfg['trn_batch_size'], steps_val),
            HistoryPlotCB('%s/history.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/wvalf2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            ReduceLROnPlateau(monitor='val_F2', factor=0.75, patience=10,
                              min_lr=1e-4, epsilon=1e-2, verbose=1, mode='max'),
            EarlyStopping(monitor='F2', min_delta=0.001, patience=5, verbose=1, mode='max'),
            TerminateOnNaN()
        ] + callbacks

        if self.cfg['trn_early_stop']:
            cb.append(EarlyStopping(monitor='val_F2', min_delta=0.001, patience=10, verbose=1, mode='max'))

        train = net.fit_generator(gen_trn, steps_per_epoch=steps_trn, epochs=self.cfg['trn_epochs'],
                                  verbose=1, callbacks=cb)

        # Return the training history.
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

                ib = np.zeros(ib_shape, dtype=np.uint8)
                tb = np.zeros(tb_shape, dtype=np.uint8)
                db = np.zeros((self.cfg['trn_batch_size'],), dtype=np.uint16)

                for bidx in range(self.cfg['trn_batch_size']):
                    db[bidx] = next(didxs_cycle)
                    img = imgs.get(str(db[bidx]))[...]
                    ib[bidx] = imresize(img, self.cfg['input_shape'])
                    tb[bidx] = tags[db[bidx]]
                    for aug in rng.choice(aug_funcs, rng.randint(0, nb_augment_max)):
                        ib[bidx] = aug(ib[bidx])

                    # import matplotlib.pyplot as plt
                    # print(didx, [TAGS[i] for i, v in enumerate(tb[bidx]) if v == 1])
                    # fig, _ = plt.subplots(1, 2)
                    # fig.axes[0].imshow(imgs.get(str(didx))[...])
                    # fig.axes[1].imshow(ib[bidx])
                    # plt.show()
                if yield_didxs:
                    yield self.cfg['preprocess_func'](ib * 1.), tb, db
                else:
                    yield self.cfg['preprocess_func'](ib * 1.), tb

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(LuckyLoser())
