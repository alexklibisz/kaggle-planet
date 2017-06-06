# Godard

import numpy as np
np.random.seed(317)

from glob import glob
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback, LearningRateScheduler
from keras.losses import kullback_leibler_divergence as KLD
from keras.losses import binary_crossentropy
from keras.applications import resnet50
from math import ceil
from os import path, mkdir, listdir
from PIL import Image
from skimage.transform import resize
from time import time
import argparse
import logging
import keras.backend as K
import pandas as pd
import pdb
import tifffile as tif


import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_ints, onehot_to_taglist, TAGS, onehot_F2, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


class SamplePlot(Callback):

    def __init__(self, batch_gen, file_name):
        super(Callback, self).__init__()
        self.logs = {}
        self.batch_gen = batch_gen
        self.file_name = file_name

    def on_train_begin(self, logs={}):
        self.logs = {}

    def on_epoch_end(self, epoch, logs={}):

        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        imgs_batch, tags_batch = next(self.batch_gen)
        tags_pred = self.model.predict(imgs_batch)

        nrows, ncols = min(len(imgs_batch), 10), 2
        fig, _ = plt.subplots(nrows, ncols, figsize=(8, nrows * 2.5))
        scatter_x = np.arange(len(TAGS))
        scatter_xticks = [t[:5] for t in TAGS]

        for idx, (img, tt, tp) in enumerate(zip(imgs_batch, tags_batch, tags_pred)):
            if idx >= nrows:
                break
            ax1, ax2 = fig.axes[idx * 2 + 0], fig.axes[idx * 2 + 1]
            ax1.axis('off')
            ax1.imshow(img[:, :, :3])
            tt, tp = tt.round().astype(np.float32), tp.round().astype(np.float32)
            ax2.scatter(scatter_x, tp - tt, marker='x')
            ax2.set_xticks(scatter_x)
            ax2.set_xticklabels(scatter_xticks, rotation='vertical')
            ax2.set_ylim(-1.1, 1.1)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['FN', 'OK', 'FP'])

        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('Epoch %d' % (epoch + 1))
        plt.savefig(self.file_name)
        plt.close()


class Godard(object):

    def __init__(self, checkpoint_name='Godard'):

        self.config = {
            'input_shape': [64, 64, 3],
            'output_shape': [17],
            'sigmoid_threshold': 0.5,
            'batch_size_tst': 500,
            'batch_size_trn': 128,
            'trn_nb_epochs': 30,
            'trn_transform': True,
            'trn_imgs_csv': 'data/train_v2.csv',
            'trn_imgs_dir': 'data/train-jpg',
            'tst_imgs_csv': 'data/sample_submission_v2.csv',
            'tst_imgs_dir': 'data/test-jpg',
            'trn_prop_trn': 0.8,
            'trn_prop_val': 0.2
        }

        self.checkpoint_name = checkpoint_name
        self.net = None
        self.rng = np.random

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.checkpoint_name, '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self):

        inputs = Input(shape=self.config['input_shape'])

        x = BatchNormalization()(inputs)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.2)(x)

        x = Flatten()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.05)(x)

        tags = Dense(self.config['output_shape'][0], activation='sigmoid')(x)
        thresh = Dense(self.config['output_shape'][0], activation='tanh')(x)

        tags_thresh = concatenate([tags, thresh], axis=1)
        tags_thresh = Lambda(lambda x: x[:, :17] + K.mean(x[:, -17:], axis=0))(tags_thresh)

        self.net = Model(inputs=inputs, outputs=tags_thresh)

        def meanyp(yt, yp):
            return K.mean(yp)

        def F2(yt, yp):
            # yt, yp = K.round(yt), K.round(yp)
            yp = K.cast(yp > self.config['sigmoid_threshold'], 'float')
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            fn = K.sum(K.clip(yt - yp, 0, 1))
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            b = 2.0
            return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))

        # def wlogloss(yt, yp):
        #     w = 1.2  # weight for false negative errors.

        #     # Binary mask for locations of false negatives.
        #     false_neg = K.clip(yt - K.cast((yp > self.config['sigmoid_threshold']), 'float'), 0, 1)

        #     # Compute the standard log loss error at each location.
        #     eps = K.epsilon()
        #     errs = -((yt * K.log(yp + eps)) + ((1 - yt) * K.log(1 - yp + eps)))

        #     # Subtract the error for all false negatives. T
        #     # Then add the error back multiplied by the weight.
        #     errs = (errs - (errs * false_neg)) + (errs * false_neg * w)

        #     return K.mean(errs)

        # self.net.compile(optimizer=Adam(1e-3), metrics=[F2, 'accuracy'], loss=wlogloss)
        self.net.compile(optimizer=Adam(1e-3), metrics=[F2, meanyp], loss='binary_crossentropy')
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

    def train(self, rng=np.random):

        imgs_idxs = np.arange(len(listdir(self.config['trn_imgs_dir'])))
        imgs_idxs = rng.choice(imgs_idxs, len(imgs_idxs))
        imgs_idxs_trn = imgs_idxs[:int(len(imgs_idxs) * self.config['trn_prop_trn'])]
        imgs_idxs_val = imgs_idxs[-int(len(imgs_idxs) * self.config['trn_prop_val']):]
        gen_trn = self.batch_gen(self.config['trn_imgs_csv'], self.config['trn_imgs_dir'], imgs_idxs_trn,
                                 self.config['trn_transform'])
        gen_val = self.batch_gen(self.config['trn_imgs_csv'], self.config['trn_imgs_dir'], imgs_idxs_val, False)

        def lrsched(epoch):
            if epoch < 15:
                return 1e-3
            elif epoch < 23:
                return 1e-4
            else:
                return 1e-5

        cb = [
            HistoryPlot('%s/history.png' % self.cpdir),
            SamplePlot(gen_trn, '%s/samples.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/weights_val_F2.hdf5' % self.cpdir, monitor='val_F2', verbose=1,
                            save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_F2', patience=3, verbose=1, mode='max'),
            # ReduceLROnPlateau(monitor='val_F2', factor=0.1, patience=5, min_lr=1e-5, epsilon=1e-2, verbose=1),
            LearningRateScheduler(lrsched)

        ]

        # Steps should run through the full training / validation set per epoch.
        nb_steps_trn = ceil(len(imgs_idxs_trn) * 1. / self.config['batch_size_trn'])
        nb_steps_val = ceil(len(imgs_idxs_val) * 1. / self.config['batch_size_trn'])

        self.net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=self.config['trn_nb_epochs'],
                               verbose=1, callbacks=cb,
                               workers=3, pickle_safe=True,
                               validation_data=gen_val, validation_steps=nb_steps_val)

    def batch_gen(self, imgs_csv, imgs_dir, imgs_idxs, transform=False, rng=np.random):

        # Read the CSV and extract image names and tags.
        df = pd.read_csv(imgs_csv)
        imgs_paths = ['%s/%s.jpg' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.strip().split(' ')) for t in df['tags'].values]

        while True:

            imgs_batch = np.zeros([self.config['batch_size_trn'], ] + self.config['input_shape'])
            tags_batch = np.zeros([self.config['batch_size_trn'], ] + self.config['output_shape'])
            _imgs_idxs = cycle(rng.choice(imgs_idxs, len(imgs_idxs)))

            for batch_idx in range(self.config['batch_size_trn']):
                img_idx = next(_imgs_idxs)
                img = self.img_path_to_img(imgs_paths[img_idx])
                if transform:
                    img = random_transforms(img, nb_min=0, nb_max=4)
                imgs_batch[batch_idx] = img
                tags_batch[batch_idx] = tagset_to_ints(tag_sets[img_idx])

            yield imgs_batch, tags_batch

    def img_path_to_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img.thumbnail(self.config['input_shape'][:2])
        img = np.asarray(img, dtype=np.float32) / 255.
        return img

    def predict(self, img_batch):
        tags_pred = self.net.predict(img_batch)
        tags_pred = (tags_pred > self.config['sigmoid_threshold']).astype(np.uint8)
        return tags_pred

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(Godard())
