# Godard

import numpy as np
import tensorflow as tf
np.random.seed(317)
tf.set_random_seed(3343)

from glob import glob
from itertools import cycle
from multiprocessing import Process
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, LambdaCallback, TerminateOnNaN
from keras.regularizers import l2
from math import ceil
from os import path, mkdir, listdir, environ
from termcolor import colored
from time import time, sleep
from tqdm import tqdm
from PIL import Image
import argparse
import logging
import keras.backend as K
import pandas as pd


import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_ints, bool_F2, onehot_to_taglist, TAGS, onehot_F2, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname

COLORS = ['red', 'yellow', 'blue']


def train_nets(nets, tags, weight_paths, verbose, config, cpdir, gpu):

    # Generator defined alongside training code.
    def batch_gen(paths_pos, paths_neg, nb_steps, augment=False):

        while True:

            # Re-shuffle paths to randomize batches.
            np.random.shuffle(paths_pos)
            np.random.shuffle(paths_neg)
            _paths_pos = cycle(paths_pos)
            _paths_neg = cycle(paths_neg)

            # Make a pass through all positive samples.
            for _ in range(nb_steps):

                imgs_batch = np.zeros([config['batch_size_trn'], ] + config['input_shape'], dtype=np.float16)
                tags_batch = np.zeros([config['batch_size_trn'], ] + config['output_shape'], dtype=np.uint8)

                for bidx in range(config['batch_size_trn']):
                    tag = round(np.random.rand())
                    img_path = next(_paths_pos) if bool(tag) else next(_paths_neg)
                    img = Image.open(img_path).convert('RGB')
                    img.thumbnail(config['input_shape'])
                    imgs_batch[bidx] = np.asarray(img) / 255.
                    tags_batch[bidx] = tag
                    if augment:
                        imgs_batch[bidx] = random_transforms(imgs_batch[bidx], nb_min=0, nb_max=10)

                yield imgs_batch, tags_batch

    if gpu is not None:
        environ['CUDA_VISIBLE_DEVICES'] = gpu

    df = pd.read_csv(config['trn_imgs_csv'])
    color = COLORS[int(gpu) % len(COLORS)]
    np.random.seed(317)
    tf.set_random_seed(317)

    for idx, (net, tag, weight_path) in enumerate(zip(nets, tags, weight_paths)):

        logger = logging.getLogger(colored('%s:%-17s (%d/%d)' % (funcname(), tag, idx, len(nets)), color))
        sys.stdout.write = lambda x: logger.info(colored(x.rstrip(), color))

        # Find all positive and negative examples for this tag.
        df_pos = df[df['tags'].str.contains(tag)]
        df_neg = df[~df['tags'].str.contains(tag)]
        name_to_path = lambda n: '%s/%s.%s' % (config['trn_imgs_dir'], n, config['img_ext'])
        paths_pos = [name_to_path(img_name) for img_name in df_pos['image_name']]
        paths_neg = [name_to_path(img_name) for img_name in df_neg['image_name']]

        # Shuffle and split for training and validation.
        np.random.shuffle(paths_pos)
        np.random.shuffle(paths_neg)
        nb_pos_trn = ceil(len(paths_pos) * config['prop_trn'])
        paths_pos_trn = paths_pos[:nb_pos_trn]
        paths_pos_val = paths_pos[nb_pos_trn:]
        nb_neg_trn = ceil(len(paths_neg) * config['prop_trn'])
        paths_neg_trn = paths_neg[:nb_neg_trn]
        paths_neg_val = paths_neg[nb_neg_trn:]

        # Number of steps to make a full pass through all positive samples.
        nb_steps_trn = ceil(len(paths_pos_trn) * 2 / config['batch_size_trn'])
        nb_steps_val = ceil(len(paths_pos_val) * 2 / config['batch_size_trn'])

        # Define generators.
        gen_trn = batch_gen(paths_pos_trn, paths_neg_trn, nb_steps_trn, config['trn_augment'])
        gen_val = batch_gen(paths_pos_val, paths_neg_val, nb_steps_val)

        mult = 2 if config['trn_augment'] else 1

        cb = [
            HistoryPlot('%s/history_%s.png' % (cpdir, tag)),
            CSVLogger('%s/history_%s.csv' % (cpdir, tag)),
            ModelCheckpoint(weight_path, monitor='F2adj', verbose=1, save_best_only=True, mode='max'),
            EarlyStopping(monitor='F2adj', patience=int(10 * mult), verbose=1, mode='max'),
            ReduceLROnPlateau(monitor='F2adj', mode='max', factor=0.5,
                              min_lr=1e-4, patience=int(3 * mult), cooldown=1, verbose=1),
            TerminateOnNaN()
        ]

        logger.info('Training for tag %s on GPU %s.\npositives=%d, negatives=%d.' %
                    (tag, gpu, len(paths_pos), len(paths_neg)))

        net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=config['trn_nb_epochs'],
                          validation_data=gen_val, validation_steps=nb_steps_val,
                          verbose=verbose, callbacks=cb, workers=2, pickle_safe=True)

    return


def create_net(config):

    inputs = Input(shape=config['input_shape'])

    ki = 'he_uniform'
    kreg = l2(1e-2)

    x = BatchNormalization(axis=3)(inputs)

    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=ki, kernel_regularizer=kreg)(x)
    x = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=ki, kernel_regularizer=kreg)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.1)(x)

    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=ki, kernel_regularizer=kreg)(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=ki, kernel_regularizer=kreg)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=ki, kernel_regularizer=kreg)(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer=ki, kernel_regularizer=kreg)(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(2, activation='softmax')(x)
    x = Lambda(lambda x: x[:, 1:])(x)

    net = Model(inputs=inputs, outputs=x)

    def F2(yt, yp):
        yp = K.round(yp)
        tp = K.sum(yt * yp)
        fp = K.sum(K.clip(yp - yt, 0, 1))
        fn = K.sum(K.clip(yt - yp, 0, 1))
        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())
        b = 2.0
        return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))

    def ypm(yt, yp):
        return K.mean(yp)

    def ytm(yt, yp):
        return K.mean(yt)

    def F2adj(yt, yp):
        return F2(yt, yp) * (1 - K.abs(K.mean(yp) - K.mean(yt)))

    net.compile(optimizer=Adam(0.0005), metrics=[F2, F2adj, 'accuracy', ytm, ypm], loss='binary_crossentropy')
    return net


class GodardEnsemble(object):

    def __init__(self, checkpoint_name='godard_ensemble'):

        self.config = {
            'input_shape': [100, 100, 3],
            'output_shape': [1, ],
            'batch_size_tst': 2400,
            'batch_size_trn': 32,
            'trn_nb_epochs': 50,
            'trn_augment': True,
            'img_ext': 'jpg',
            'trn_imgs_csv': 'data/train_v2.csv',
            'trn_imgs_dir': 'data/train-jpg',
            'tst_imgs_csv': 'data/sample_submission_v2.csv',
            'tst_imgs_dir': 'data/test-jpg',
            'prop_trn': 0.7,
            'prop_val': 0.3,
        }

        self.checkpoint_name = checkpoint_name
        self.weights_path = '%s/weights_all.txt' % self.cpdir
        self.net = None

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.checkpoint_name, '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self, weights_path=None):

        # Networks stored in order corresponding to the tags.
        self.nets = [create_net(self.config) for t in TAGS]

        # Single summary and plot.
        self.nets[-1].summary()
        plot_model(self.nets[-1], to_file='%s/net.png' % self.cpdir)

        # Load weights if given.
        if weights_path is not None:
            weight_paths = [l.strip() for l in open(weights_path)]
            for net, p in zip(nets, weight_paths):
                net.load_weights(p)

    def train(self):

        logger = logging.getLogger(funcname())

        # Save file with weight paths checkpointed during training.
        weight_paths = ['%s/weights_%s.hdf5' % (self.cpdir, t) for t in TAGS]
        f = open(self.weights_path, 'w')
        f.write('\n'.join(weight_paths))
        f.close()

        # Split up training across available GPUs / processes if possible.
        gpu_key = 'CUDA_VISIBLE_DEVICES'
        gpus = environ[gpu_key].split(',') if gpu_key in environ else ['0']

        if len(gpus) == 1:
            train_nets(self.nets, TAGS, weight_paths, 1, self.config, self.cpdir, gpus[0])

        else:
            nb_models = len(TAGS)
            nb_models_per_gpu = ceil(nb_models / len(gpus))
            logger.info('Training %d nets on %d GPUs.' % (nb_models, len(gpus)))
            processes = []
            for gpu, idx0 in zip(gpus, range(0, nb_models, nb_models_per_gpu)):
                idx1 = min(idx0 + nb_models_per_gpu, nb_models)
                args = (self.nets[idx0:idx1], TAGS[idx0:idx1], weight_paths[idx0:idx1], 2, self.config, self.cpdir, gpu)
                p = Process(target=train_nets, args=args)
                p.start()
                processes.append(p)

            # Wait to exit.
            for p in processes:
                p.join()

        logger.info('Training complete.')

    def _get_img_by_path(self, img_path):
        return

    def predict(self, img_names, thresholds=None):
        return

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(GodardEnsemble())
