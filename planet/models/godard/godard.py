# Godard

import numpy as np
np.random.seed(317)

from glob import glob
from hyperopt import hp, fmin, tpe, space_eval
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
import tifffile as tif


import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_ints, bool_F2, onehot_to_taglist, TAGS, onehot_F2, random_transforms
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
            'input_shape': [96, 96, 3],
            'output_shape': [17],
            'sigmoid_threshold': 0.2,
            'batch_size_tst': 2400,
            'batch_size_trn': 128,
            'trn_nb_epochs': 30,
            'trn_transform': True,
            'trn_imgs_csv': 'data/train_v2.csv',
            'trn_imgs_dir': 'data/train-jpg',
            'tst_imgs_csv': 'data/sample_submission_v2.csv',
            'tst_imgs_dir': 'data/test-jpg',
            'trn_prop_trn': 0.8,
            'trn_prop_val': 0.2,
            'optimize_individually': True
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

        x = BatchNormalization(input_shape=self.config['input_shape'])(inputs)

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
        labels = Dense(self.config['output_shape'][0], activation='sigmoid')(x)

        self.net = Model(inputs=inputs, outputs=labels)

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

        self.net.compile(optimizer=Adam(1e-3), metrics=[F2, 'accuracy'], loss='binary_crossentropy')
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
            ModelCheckpoint('%s/weights_val_acc.hdf5' % self.cpdir, monitor='val_acc', verbose=1,
                            save_best_only=True, mode='max', save_weights_only=True),
            ModelCheckpoint('%s/weights_val_loss.hdf5' % self.cpdir, monitor='val_loss', verbose=1,
                            save_best_only=True, mode='min', save_weights_only=True),
            EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min'),
            LearningRateScheduler(lrsched)
        ]

        # Steps should run through the full training / validation set per epoch.
        nb_steps_trn = ceil(len(imgs_idxs_trn) * 1. / self.config['batch_size_trn'])
        nb_steps_val = ceil(len(imgs_idxs_val) * 1. / self.config['batch_size_trn'])

        self.net.fit_generator(gen_trn, steps_per_epoch=nb_steps_trn, epochs=self.config['trn_nb_epochs'],
                               verbose=1, callbacks=cb, workers=3, pickle_safe=True, max_q_size=100,
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
                    img = random_transforms(img, nb_min=0, nb_max=5)
                imgs_batch[batch_idx] = img
                tags_batch[batch_idx] = tagset_to_ints(tag_sets[img_idx])

            yield imgs_batch, tags_batch

    def img_path_to_img(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img.thumbnail(self.config['input_shape'][:2])
        img = np.asarray(img, dtype=np.float32) / 255
        return img

    def predict(self, img_batch, thresholds=None):
        if thresholds is None:
            thresholds = np.ones((17,)) * self.config['sigmoid_threshold']
            # Optimized individually for val_acc weights: thresholds = np.ones((17,)) * [0.2057822534120738, 0.17904424176966482, 0.16173641494347635, 0.17749436555197307, 0.1718912592676496, 0.35533438982531207, 0.1267769380503393, 0.1484117630813867, 0.2545285746778627, 0.20573219907959894, 0.25606849177782665, 0.05334252213115975, 0.22231628488975974, 0.172388062350327, 0.17376665737489883, 0.21450229219753403, 0.18220747827061168]
            # Optimized all at once: thresholds = np.ones((17,)) * [0.3949744842738293, 0.35646670896414034, 0.2354632378506592, 0.21900141790417665, 0.09663219418724556, 0.16309886126905748, 0.09862582812410145, 0.40505818388873943, 0.29467840611694573, 0.5340765231021278, 0.44163066996551353, 0.17753985846873613, 0.27459307412361633, 0.24140090011869653, 0.05191553521314547, 0.052724892922606015, 0.4036503445039251]

        tags_pred = self.net.predict(img_batch)
        # Alternative thresholds from https://goo.gl/hWfj8c.
        # godard_thresh_vals = [0.245, 0.1375, 0.2225, 0.19, 0.0475, 0.2375, 0.12,
        #                       0.0875, 0.265, 0.2175, 0.1925, 0.1625, 0.2625, 0.21, 0.14, 0.085, 0.205]
        tags_pred = (tags_pred > thresholds).astype(np.uint8)
        return tags_pred

    def optimize(self):
        df = pd.read_csv(self.config['trn_imgs_csv'])
        imgs_paths = ['%s/%s.jpg' % (self.config['trn_imgs_dir'], n) for n in df['image_name'].values]
        tag_sets = [tagset_to_ints(set(t.strip().split(' '))) for t in df['tags'].values]

        imgs_batch = np.zeros([self.config['batch_size_tst'], ] + self.config['input_shape'])

        if self.config['optimize_individually']:
            thresholds = [0.2] * self.config['output_shape'][0]
            for i in range(len(thresholds)):
                def objective(threshold):
                    thresholds[i] = threshold
                    scores = self.get_f2_scores(imgs_paths, tag_sets, imgs_batch, thresholds)
                    return -np.mean(scores)
                space = hp.uniform(str(i), 0, 1)

                # maximize F2 by optimizing this threshold
                thresholds[i] = fmin(objective, space, algo=tpe.suggest, max_evals=10)[str(i)]
                print(i, thresholds)
            print(thresholds)
        else:
            space = [hp.uniform(str(i), 0, 1) for i in range(self.config['output_shape'][0])]
            def objective(thresholds):
                scores = self.get_f2_scores(imgs_paths, tag_sets, imgs_batch, thresholds)
                return -np.mean(scores)
            # maximize F2 by optimizing all thresholds
            best = fmin(objective, space, algo=tpe.suggest, max_evals=200)
            print(best)
            print(space_eval(space, best))

    def get_f2_scores(self, imgs_paths, tags_true, imgs_batch, thresholds):
        batch_size = len(imgs_batch)
        F2_scores = []
        print(thresholds)
        for idx in range(0, len(imgs_paths), batch_size):
            for _, img_path in enumerate(imgs_paths[idx:idx + batch_size]):
                imgs_batch[_] = self.img_path_to_img(img_path)

            # Make predictions, compute F2 and store it.
            tags_pred = self.predict(imgs_batch, thresholds)
            for tt, tp in zip(tags_true[idx:idx + batch_size], tags_pred):
                F2_scores.append(bool_F2(tt, tp))

            # Progress...
            print('%d/%d F2 running = %lf, F2 batch = %lf' %
                        (idx, len(imgs_paths), np.mean(F2_scores), np.mean(F2_scores[idx:])))
        return F2_scores


if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(Godard())

