# VGGNet
import numpy as np
np.random.seed(317)

from glob import glob
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate
from keras.utils import plot_model
from os import path, mkdir
import keras.backend as K
import pandas as pd

import sys
sys.path.append('.')
from planet.utils.data_utils import get_imgs_lbls


class VGGNet(object):

    def __init__(self, checkpoint_name='VGGNet'):

        self.config = {
            'img_shape': [256, 256, 4],
            'input_shape': [256, 256, 4],
            'output_shape': [17, 2],
            'batch_size': 10
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

    def load_data(self, imgs_dir='data/train-tif-sample', csv_path='data/train.csv'):
        self.imgs, self.lbls = get_imgs_lbls(imgs_dir, csv_path)
        return

    def create_net(self):

        x = inputs = Input(shape=self.config['input_shape'])

        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = Conv2D(512, 3, padding='same', activation='relu')(x)
        x = MaxPooling2D(2, strides=2)(x)

        x = Flatten()(x)
        x = Dense(2048)(x)
        dense_all = x = Dense(2048)(x)

        activations = []
        for n in range(self.config['output_shape'][0]):
            activations.append(Dense(2, activation='softmax')(dense_all))

        x = concatenate(activations, axis=-1)
        outputs = x = Reshape(self.config['output_shape'])(x)

        self.net = Model(inputs=inputs, outputs=outputs)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        self.net.compile(loss='binary_crossentropy', optimizer=Adam(0.0005))

        return

    def train(self):
        self.net.fit(self.imgs, self.lbls, batch_size=10, verbose=1, epochs=10, shuffle=True)
        return


if __name__ == "__main__":

    model = VGGNet()
    model.create_net()
    model.load_data()
    model.train()
