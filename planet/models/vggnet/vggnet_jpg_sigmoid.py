# VGGNet
import numpy as np
np.random.seed(317)

from glob import glob
from itertools import cycle
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, BatchNormalization, Flatten, Dropout, Dense
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback
from keras.losses import kullback_leibler_divergence
from math import ceil
from os import path, mkdir, listdir
from skimage.transform import resize
from scipy.misc import imread, imsave
from time import time
import argparse
import logging
import keras.backend as K
import pandas as pd
import tifffile as tif

import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_ints, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


class VGGNet(object):

    def __init__(self, checkpoint_name='VGGNet'):

        self.config = {
            'image_shape': [256, 256, 3],
            'input_shape': [224, 224, 3],
            'output_shape': [17, ],
            'batch_size': 60,
            'trn_steps': 680,
            'trn_nb_epochs': 200,
            'trn_transform': True,
            'trn_imgs_csv': 'data/train_v2.csv',
            'trn_imgs_dir': 'data/train-jpg',
            'tst_imgs_csv': 'data/sample_submission_v2.csv',
            'tst_imgs_dir': 'data/test-jpg'
        }

        self.checkpoint_name = checkpoint_name
        self.imgs = []
        self.lbls = []
        self.net = None
        self.rng = np.random

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.checkpoint_name, '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self):

        x = inputs = Input(shape=self.config['input_shape'])
        vgg = VGG19(include_top=False, input_tensor=x)

        outputs = Flatten()(vgg.output)
        outputs = Dense(self.config['output_shape'][0], activation='sigmoid')(outputs)

        def true_pos(yt, yp):
            return K.sum(K.round(yt)) / K.sum(K.clip(yt, 1, 1))

        def pred_pos(yt, yp):
            return K.sum(K.round(yp)) / K.sum(K.clip(yt, 1, 1))

        def F2(yt, yp):
            yt, yp = K.round(yt), K.round(yp)
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            fn = K.sum(K.clip(yt - yp, 0, 1))
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            b = 2.0
            return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))

        self.net = Model(inputs, outputs)
        self.net.compile(optimizer=Adam(0.001), loss='binary_crossentropy',
                         metrics=['binary_accuracy', F2, true_pos, pred_pos])
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)
        return

    def train(self):

        batch_gen = self.train_batch_gen(self.config['trn_imgs_csv'], self.config[
                                         'trn_imgs_dir'], self.config['trn_transform'])

        cb = [
            HistoryPlot('%s/history.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/loss.weights' % self.cpdir, monitor='loss', verbose=1,
                            save_best_only=True, mode='min', save_weights_only=True),
            ModelCheckpoint('%s/dice_coef.weights' % self.cpdir, monitor='dice_coef',
                            verbose=1, save_best_only=True, mode='max', save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, epsilon=0.005, verbose=1, mode='min'),
            EarlyStopping(monitor='loss', min_delta=0.01, patience=10, verbose=1, mode='min')
        ]

        self.net.fit_generator(batch_gen, steps_per_epoch=self.config['trn_steps'], verbose=1, callbacks=cb,
                               epochs=self.config['trn_nb_epochs'], workers=2, pickle_safe=True)

        return

    def get_mean_img(self, imgs_paths, mean_img_path):
        '''Compute the mean image from the given paths and save it to the given path.'''
        logger = logging.getLogger(funcname())
        if not path.exists(mean_img_path):
            mean_img = np.zeros(self.config['image_shape'], dtype=np.float32)
            for idx, img_path in enumerate(imgs_paths):
                mean_img += imread(img_path, mode='RGB').astype(np.float32) / len(imgs_paths)
                if idx % 1000 == 0:
                    logger.info('%d/%d' % (idx, len(imgs_paths)))
            imsave(mean_img_path, mean_img)
        return imread(mean_img_path)

    def train_batch_gen(self, imgs_csv, imgs_dir, transform):

        logger = logging.getLogger(funcname())

        # Read the CSV and extract image names and tags.
        df = pd.read_csv(imgs_csv)
        imgs_paths = ['%s/%s.jpg' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.strip().split(' ')) for t in df['tags'].values]

        # Compute the mean image for pre-processing.
        mean_img = self.get_mean_img(imgs_paths, '%s/mean_img_trn.jpg' % self.cpdir)
        mean_img = mean_img.astype(np.float32) / 255.
        mean_img_mean = np.mean(mean_img)
        img_preprocess = lambda img: img.astype(np.float32) / 255. - mean_img_mean

        while True:

            imgs_batch = np.zeros([self.config['batch_size'], ] + self.config['input_shape'])
            tags_batch = np.zeros([self.config['batch_size'], ] + self.config['output_shape'])
            random_idxs = cycle(np.random.choice(np.arange(len(imgs_paths)), len(imgs_paths)))

            for batch_idx in range(self.config['batch_size']):
                data_idx = next(random_idxs)
                img = imread(imgs_paths[data_idx], mode='RGB')
                img = img_preprocess(img)
                img = resize(img, self.config['input_shape'], preserve_range=True, mode='constant')
                if transform:
                    img = random_transforms(img)
                imgs_batch[batch_idx] = img
                tags_batch[batch_idx] = tagset_to_ints(tag_sets[data_idx])

            yield imgs_batch, tags_batch

    def predict(self, img_batch):

        # Get the mean image
        imgs_paths = listdir(self.config['trn_imgs_dir'])
        mean_img_path = '%s/mean_img_trn.jpg' % self.cpdir
        mean_img = self.get_mean_img(imgs_paths, mean_img_path).astype(np.float32) / 255.
        mean_img_mean = np.mean(mean_img)
        img_preprocess = lambda img: img.astype(np.float32) / 255. - mean_img_mean

        for idx in range(len(img_batch)):
            img_batch[idx] = img_preprocess(img_batch[idx])

        tags_pred = self.net.predict(img_batch)
        tags_pred = tags_pred.round().astype(np.uint8)
        return tags_pred

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model = VGGNet()
    model_runner(model)
