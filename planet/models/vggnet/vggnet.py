# VGGNet
import numpy as np
np.random.seed(317)

from glob import glob
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from os import path, mkdir
import argparse
import logging
import keras.backend as K
import pandas as pd
import tifffile as tif

import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_onehot
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


class VGGNet(object):

    def __init__(self, checkpoint_name='VGGNet'):

        self.config = {
            'img_shape': [256, 256, 4],
            'input_shape': [256, 256, 4],
            'output_shape': [17, 2],
            'batch_size': 20,
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
        x = Dropout(0.1)(x)

        x = MaxPooling2D(2, strides=2)(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Conv2D(64, 3, padding='same', activation='relu')(x)
        x = Dropout(0.1)(x)

        x = MaxPooling2D(2, strides=2)(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Conv2D(128, 3, padding='same', activation='relu')(x)
        x = Dropout(0.1)(x)

        x = MaxPooling2D(2, strides=2)(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Conv2D(256, 3, padding='same', activation='relu')(x)
        x = Dropout(0.1)(x)

        x = MaxPooling2D(2, strides=2)(x)
        conv_flat = Flatten()(x)

        # Each tag has its own mini dense classifier operating on the conv outputs.
        classifiers = []
        for n in range(self.config['output_shape'][0]):
            x = Dense(22, activation='relu')(conv_flat)
            x = Dropout(0.1)(x)
            classifiers.append(Dense(2, activation='softmax')(x))

        # Concatenate and reshape the classifier outputs.
        x = concatenate(classifiers, axis=-1)
        tags = x = Reshape(self.config['output_shape'], name='tags')(x)

        # Secondary output outputs the normalized probability distribution to be measured
        # against the true probability distribution via KL-divergence loss.
        def to_distribution(tags_onehot):
            tags = K.cast(K.argmax(tags_onehot, axis=2), 'float32')
            return tags / (K.sum(tags) + K.epsilon())

        dist = Lambda(to_distribution, name='dist')(tags)

        self.net = Model(inputs=inputs, outputs=[tags, dist])

        def dbg_act(yt, yp):
            '''Return the mean activation of each classifier (should be 0.5)'''
            m = K.mean(yp, axis=2)
            return K.mean(m)

        def dice_coef(yt, yp):
            '''Dice coefficient from VNet paper.'''
            t, p = yt[:, :, 1], yp[:, :, 1]
            nmr = 2 * K.sum(t * p)
            dnm = K.sum(t**2) + K.sum(p**2) + K.epsilon()
            return nmr / dnm

        def dice_loss(yt, yp):
            return 1 - dice_coef(yt, yp)

        self.net.compile(optimizer=Adam(0.001),
                         metrics={'tags': [dbg_act, dice_coef]},
                         loss={'tags': dice_loss,
                               'dist': 'kullback_leibler_divergence'},
                         loss_weights={'tags': 1., 'dist': 0.4})

        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

        return

    def train(self):

        batch_gen = self.train_batch_gen('data/train.csv', 'data/train-tif', self.config['batch_size'])

        cb = [
            HistoryPlot('%s/history.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/loss.weights' % self.cpdir, monitor='loss', verbose=1, save_best_only=True, mode='min'),
            ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10, verbose=1, mode='min'),
        ]

        self.net.fit_generator(batch_gen, steps_per_epoch=1000, verbose=1, callbacks=cb,
                               epochs=self.config['nb_epochs'], workers=3, pickle_safe=True, max_q_size=100)

        return

    def train_batch_gen(self, csv_path='data/train.csv', imgs_dir='data/train-tif', batch_size=32, rng=np.random):

        logger = logging.getLogger(funcname())

        # Helpers.
        scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
        onehot_to_distribution = lambda x: np.argmax(x, axis=1) / np.sum(np.argmax(x, axis=1))

        # Read the CSV and error-check contents.
        df = pd.read_csv(csv_path)
        img_names = ['%s/%s.tif' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.strip().split(' ')) for t in df['tags'].values]

        for img_name, tag_set in zip(img_names, tag_sets):
            assert path.exists(img_name), img_name
            assert len(tag_set) > 0, tag_set

        while True:

            # New batches at each iteration to prevent over-writing previous batch before it's used.
            imgs_batch = np.zeros([batch_size, ] + self.config['input_shape'], dtype=np.float32)
            tags_batch = np.zeros([batch_size, ] + self.config['output_shape'], dtype=np.uint8)
            dist_batch = np.zeros([batch_size, ] + self.config['output_shape'][:1], dtype=np.float32)

            # Sample *batch_size* random rows and build the batches.
            for batch_idx in range(batch_size):
                data_idx = rng.randint(0, len(img_names))

                try:
                    imgs_batch[batch_idx] = scale(tif.imread(img_names[data_idx]))
                    tags_batch[batch_idx] = tagset_to_onehot(tag_sets[data_idx])
                    dist_batch[batch_idx] = onehot_to_distribution(tags_batch[batch_idx])
                except Exception as e:
                    logger.error(img_names[data_idx])
                    logger.error(str(e))
                    pass

            # TODO: Image augmentation.

            yield imgs_batch, [tags_batch, dist_batch]


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('VGGNet')

    parser = argparse.ArgumentParser(description='VGGNet Model.')
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    parser_train = sub.add_parser('train', help='training')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-c', '--config', help='config file')
    parser_train.add_argument('-w', '--weights', help='network weights')

    # Prediction / submission.
    parser_predict = sub.add_parser('predict', help='make predictions')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('-c', '--config', help='config file')
    parser_predict.add_argument('-w', '--weights', help='network weights', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict']

    model = VGGNet()
    model.create_net()

    def load_weights():
        if args['weights'] is not None:
            logger.info('Loading network weights from %s.' % args['weights'])
            model.net.load_weights(args['weights'])

    if args['which'] == 'train':
        load_weights()
        model.train()

    elif args['which'] == 'predict':
        logger.info('TODO')
