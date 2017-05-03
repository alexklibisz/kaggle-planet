# VGGNet
import numpy as np
np.random.seed(317)

from glob import glob
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from os import path, mkdir
import keras.backend as K
import pandas as pd
import tifffile as tif

import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_onehot


class VGGNet(object):

    def __init__(self, checkpoint_name='VGGNet'):

        self.config = {
            'img_shape': [256, 256, 4],
            'input_shape': [256, 256, 4],
            'output_shape': [17, 2],
            'batch_size': 25,
            'nb_epochs': 100,
        }

        self.checkpoint_name = checkpoint_name
        self.imgs = []
        self.lbls = []
        self.net = None

    @property
    def cpdir(self):
        cpdir = 'checkpoints/%s_%s/' % (self.checkpoint_name, '_'.join([str(x) for x in self.config['input_shape']]))
        if not path.exists(cpdir):
            mkdir(cpdir)
        return cpdir

    def create_net(self):

        x = inputs = Input(shape=self.config['input_shape'])

        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = Conv2D(32, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        conv_flat = Flatten()(x)

        # Each tag has its own mini dense classifier operating on the conv outputs.
        classifiers = []
        for n in range(self.config['output_shape'][0]):
            x = Dense(32, activation='relu')(conv_flat)
            classifiers.append(Dense(2, activation='softmax')(x))

        # Concatenate and reshape the classifier outputs.
        x = concatenate(classifiers, axis=-1)
        tags_out = x = Reshape(self.config['output_shape'])(x)

        self.net = Model(inputs=inputs, outputs=tags_out)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        # TODO: Secondary output returns the normalized class distribution with KL-divergence loss.

        def dbg_act(yt, yp):
            '''Return the mean activation of each classifier (should be 0.5)'''
            m = K.mean(yp, axis=2)
            return K.mean(m)

        def dice_coef(yt, yp):
            t, p = yt[:, :, 1], yp[:, :, 1]
            nmr = 2 * K.sum(t * p)
            dnm = K.sum(t**2) + K.sum(p**2) + K.epsilon()
            return nmr / dnm

        def dice_loss(yt, yp):
            return 1 - dice_coef(yt, yp)

        self.net.compile(optimizer=Adam(0.0005), metrics=[dbg_act, dice_coef],
                         # loss='binary_crossentropy'
                         loss=dice_loss)

        return

    def train(self):

        batch_gen = self.train_batch_gen('data/train.csv', 'data/train-tif', self.config['batch_size'])

        cb = [
            ModelCheckpoint('%s/loss.weights' % self.cpdir, monitor='loss', verbose=1, save_best_only=True)
        ]

        self.net.fit_generator(batch_gen, steps_per_epoch=100, verbose=1, callbacks=cb,
                               epochs=self.config['nb_epochs'], workers=3, pickle_safe=True)

        return

    def train_batch_gen(self, csv_path='data/train.csv', imgs_dir='data/train-tif', batch_size=32, rng=np.random):

        # Helpers.
        scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1

        # Read the CSV.
        df = pd.read_csv(csv_path)
        img_names = ['%s/%s.tif' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.split(' ')) for t in df['tags'].values]

        while True:

            # New batches at each iteration to prevent over-writing previous batch before it's used.
            img_batch = np.zeros([batch_size, ] + self.config['input_shape'], dtype=np.float32)
            tag_batch = np.zeros([batch_size, ] + self.config['output_shape'], dtype=np.uint8)

            # Sample *batch_size* random rows and build the batches.
            for batch_idx in range(batch_size):
                data_idx = rng.randint(0, len(img_names))
                img_batch[batch_idx] = scale(tif.imread(img_names[data_idx]))
                tag_batch[batch_idx] = tagset_to_onehot(tag_sets[data_idx])

            # TODO: Image augmentation.

            yield img_batch, tag_batch


if __name__ == "__main__":

    model = VGGNet()
    model.create_net()
    model.train()
