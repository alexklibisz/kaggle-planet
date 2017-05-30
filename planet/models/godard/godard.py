# Godard

import numpy as np
np.random.seed(317)

from glob import glob
from itertools import cycle
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping, Callback
from keras.losses import kullback_leibler_divergence as KLD
from keras.losses import binary_crossentropy
from keras.applications import resnet50
from os import path, mkdir
from skimage.transform import resize
from time import time
import argparse
import logging
import keras.backend as K
import pandas as pd
from scipy.misc import imread
import tifffile as tif
from sklearn.model_selection import train_test_split
from math import ceil
from random import shuffle

import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_ints, onehot_to_taglist, TAGS, onehot_F2, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname

from PIL import Image

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
            'batch_size': 300,  # Big boy GPU.
            # 'nb_epochs': 200,
            'validation_split_size': 0.2,
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

        inputs = Input(shape=self.config['input_shape'])

        x = BatchNormalization(input_shape=self.config['input_shape'])(inputs)

        x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(256, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=2)(x)
        x = Dropout(0.25)(x)

        x = Flatten()(x)

        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        labels = Dense(self.config['output_shape'][0], activation='sigmoid', name='labels')(x)

        self.net = Model(inputs=inputs, outputs=labels)


        # self.net.compile(optimizer=Adam(0.001), metrics=[F2, dice_coef, kl_loss], loss=custom_loss)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

    def train(self):
        train_batch_gen, validate_batch_gen = self.get_batch_gens(self.config['trn_imgs_csv'], 
                                                                self.config['trn_imgs_dir'], 
                                                                self.config['validation_split_size'])

        for i in train_batch_gen:
            # import pdb
            # pdb.set_trace()
            break

	# Make sure the history doesn't get cleared out each training run
        history_plot = HistoryPlot('%s/history.png' % self.cpdir)
        def empty(self):
            pass
        history_plot.on_train_begin = empty

        cb = [
            SamplePlot(train_batch_gen, '%s/samples.png' % self.cpdir),
            history_plot,
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/loss.weights' % self.cpdir, monitor='val_acc', verbose=1, # monitor was 'loss'
                            save_best_only=True, mode='auto', save_weights_only=True), # mode was 'min'
            # ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, epsilon=0.005, verbose=1, mode='min'),
            EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto') # monitor='loss', patience=15, mode='min', min_delta=0.01
        ]

        def F2(yt, yp):
            yt, yp = K.round(yt), K.round(yp)
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            fn = K.sum(K.clip(yt - yp, 0, 1))
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            b = 2.0
            return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))


        epochs_arr = [10, 5, 5]
        learn_rates = [0.001, 0.0001, 0.00001]

        train_steps = ceil((1 - self.config['validation_split_size']) * self.num_examples/self.config['batch_size'])
        validation_steps = ceil(self.config['validation_split_size'] * self.num_examples/self.config['batch_size'])
        for learn_rate, epochs in zip(learn_rates, epochs_arr):
            self.net.compile(optimizer=Adam(learn_rate), metrics=[F2, 'accuracy'], loss='binary_crossentropy')
            self.net.fit_generator(train_batch_gen, steps_per_epoch=train_steps, verbose=1, callbacks=cb,
                                   epochs=epochs, workers=3, pickle_safe=True, max_q_size=100,
                                   validation_data=validate_batch_gen, validation_steps=validation_steps)

        return

    def get_batch_gens(self, csv_path, imgs_dir, validation_split_size):

        logger = logging.getLogger(funcname())

        # Read the CSV and error-check contents.
        df = pd.read_csv(csv_path)
        img_names = ['%s/%s.jpg' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.strip().split(' ')) for t in df['tags'].values]

        # Error check.
        for img_name, tag_set in zip(img_names, tag_sets):
            assert path.exists(img_name), img_name
            assert len(tag_set) > 0, tag_set

        # Record the number of images
        self.num_examples = len(img_names)

        imgs_train, imgs_test, tags_train, tags_test = train_test_split(img_names, tag_sets, test_size=validation_split_size, random_state=42)
        
        return self.batch_gen(imgs_train, tags_train), self.batch_gen(imgs_test, tags_test)

    def batch_gen(self, img_names, tag_sets):
        # Helper
        # scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1

        data_idxs = list(range(len(img_names)))
        batch_idx = 0
        while True:
            
            # New batches at each iteration to prevent over-writing previous batch before it's used.
            imgs_batch = np.zeros([self.config['batch_size'], ] + self.config['input_shape'], dtype=np.float32)
            tags_batch = np.zeros([self.config['batch_size'], ] + self.config['output_shape'], dtype=np.uint8)

            # Run through the data in a random order
            shuffle(data_idxs)
            for data_idx in data_idxs:
                img = Image.open(img_names[data_idx])
                img.thumbnail(self.config['input_shape'][:2])
                img = np.array(img.convert("RGB"), dtype=np.float32) / 255
                # img = resize(imread(img_names[data_idx], mode='RGB'), self.config[
                #              'input_shape'][:2], preserve_range=True, mode='constant')
                if self.config['trn_transform']:
                    imgs_batch[batch_idx] = random_transforms(img, nb_min=0, nb_max=2)
                    # imgs_batch[batch_idx] = scale(random_transforms(img, nb_min=0, nb_max=2))
                else:
                    imgs_batch[batch_idx] = scale(img)

                tags_batch[batch_idx] = tagset_to_ints(tag_sets[data_idx])

                batch_idx += 1

                # Yield batch if it's big enough
                if batch_idx >= self.config['batch_size']:
                    batch_idx = 0
                    yield imgs_batch, tags_batch
                    break


    def predict(self, img_batch):

        scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1

        for idx in range(len(img_batch)):
            img_batch[idx] = scale(img_batch[idx])

        tags_pred = self.net.predict(img_batch)
        tags_pred = tags_pred.round().astype(np.uint8)

        # Convert from onehot to an array of bools
        tags_pred = tags_pred[:, :, 1]
        return tags_pred

if __name__ == "__main__":
    from planet.model_runner import model_runner
    model_runner(Godard())
