# ResNet50

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
import tifffile as tif

import sys
sys.path.append('.')
from planet.utils.data_utils import tagset_to_onehot, onehot_to_taglist, TAGS, onehot_F2, random_transforms
from planet.utils.keras_utils import HistoryPlot
from planet.utils.runtime import funcname


class SamplePlot(Callback):  # copied unmodified from indiconv.py and vggnet.py - 5/16/2017

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
        tags_pred = model.net.predict(imgs_batch)

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
            ax2.scatter(scatter_x, np.argmax(tp, axis=1) - np.argmax(tt, axis=1), marker='x')
            ax2.set_xticks(scatter_x)
            ax2.set_xticklabels(scatter_xticks, rotation='vertical')
            ax2.set_ylim(-1.1, 1.1)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['FN', 'OK', 'FP'])

        plt.subplots_adjust(hspace=0.5)
        plt.suptitle('Epoch %d' % (epoch + 1))
        plt.savefig(self.file_name)
        plt.close()


class ResNet50(object):

    def __init__(self, checkpoint_name='ResNet50'):

        self.config = {
            'input_shape': [224, 224, 4],
            'output_shape': [17, 2],
            'batch_size': 50,  # Big boy GPU.
            'nb_epochs': 200,
            'trn_transform': True,
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
        if tuple(self.config['input_shape']) != (224, 224, 4):
            print("ResNet50 has special restrictions on the input shape. Only remove this sys.exit call if you know what you're doing.")
            sys.exit(1)

        inputs = Input(shape=self.config['input_shape'])
        res = resnet50.ResNet50(include_top=False, weights=None, input_tensor=inputs)

        # Add a classifier for each class instead of a single classifier.
        res_out = Flatten()(res.output)
        res_out = Dropout(0.1)(res_out)
        # res_out = Dense(512, activation='relu')(x)

        classifiers = []
        for n in range(self.config['output_shape'][0]):
            classifiers.append(Dense(2, activation='softmax')(res_out))

        # Concatenate classifiers and reshape to match output shape.
        x = concatenate(classifiers, axis=-1)
        labels = x = Reshape(self.config['output_shape'], name='labels')(x)

        self.net = Model(inputs=res.input, outputs=labels)

        def F2(yt, yp):
            yt, yp = K.cast(K.argmax(yt, axis=2), 'float32'), K.cast(K.argmax(yp, axis=2), 'float32')
            tp = K.sum(yt * yp)
            fp = K.sum(K.clip(yp - yt, 0, 1))
            fn = K.sum(K.clip(yt - yp, 0, 1))
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            b = 2.0
            return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))

        def dice_coef(yt, yp):
            '''Dice coefficient from VNet paper.'''
            yt, yp = yt[:, :, 1], yp[:, :, 1]
            nmr = 2 * K.sum(yt * yp)
            dnm = K.sum(yt**2) + K.sum(yp**2) + K.epsilon()
            return nmr / dnm

        def dice_loss(yt, yp):
            return 1 - dice_coef(yt, yp)

        def kl_loss(yt, yp):
            return KLD(K.flatten(yt) / K.sum(yt), K.flatten(yp) / K.sum(yp))

        def custom_loss(yt, yp):
            return binary_crossentropy(yt, yp)

        self.net.compile(optimizer=Adam(0.001), metrics=[F2, dice_coef, kl_loss], loss=custom_loss)
        self.net.summary()
        plot_model(self.net, to_file='%s/net.png' % self.cpdir)

    def train(self):

        batch_gen = self.train_batch_gen('data/train_v2.csv', 'data/train-tif-v2')

        cb = [
            SamplePlot(batch_gen, '%s/samples.png' % self.cpdir),
            HistoryPlot('%s/history.png' % self.cpdir),
            CSVLogger('%s/history.csv' % self.cpdir),
            ModelCheckpoint('%s/loss.weights' % self.cpdir, monitor='loss', verbose=1,
                            save_best_only=True, mode='min', save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2, epsilon=0.005, verbose=1, mode='min'),
            EarlyStopping(monitor='loss', min_delta=0.01, patience=15, verbose=1, mode='min')
        ]

        self.net.fit_generator(batch_gen, steps_per_epoch=500, verbose=1, callbacks=cb,
                               epochs=self.config['nb_epochs'], workers=1, pickle_safe=True, max_q_size=100)

        return

    def train_batch_gen(self, csv_path='data/train_v2.csv', imgs_dir='data/train-tif-v2'):

        logger = logging.getLogger(funcname())

        # Helpers.
        scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
        onehot_to_distribution = lambda x: np.argmax(x, axis=1) / np.sum(np.argmax(x, axis=1))

        # Read the CSV and error-check contents.
        df = pd.read_csv(csv_path)
        img_names = ['%s/%s.tif' % (imgs_dir, n) for n in df['image_name'].values]
        tag_sets = [set(t.strip().split(' ')) for t in df['tags'].values]

        # Error check.
        for img_name, tag_set in zip(img_names, tag_sets):
            assert path.exists(img_name), img_name
            assert len(tag_set) > 0, tag_set

        while True:

            # New batches at each iteration to prevent over-writing previous batch before it's used.
            imgs_batch = np.zeros([self.config['batch_size'], ] + self.config['input_shape'], dtype=np.float32)
            tags_batch = np.zeros([self.config['batch_size'], ] + self.config['output_shape'], dtype=np.uint8)

            # Sample *self.config['batch_size']* random rows and build the batches.
            for batch_idx in range(self.config['batch_size']):
                data_idx = self.rng.randint(0, len(img_names))

                try:
                    img = resize(tif.imread(img_names[data_idx]), self.config[
                                 'input_shape'][:2], preserve_range=True, mode='constant')
                    if self.config['trn_transform']:
                        imgs_batch[batch_idx] = scale(random_transforms(img, nb_min=0, nb_max=2))
                    else:
                        imgs_batch[batch_idx] = scale(img)
                    tags_batch[batch_idx] = tagset_to_onehot(tag_sets[data_idx])
                except Exception as e:
                    pass

            yield imgs_batch, tags_batch

    def predict(self, img_batch):

        scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1

        for idx in range(len(img_batch)):
            img_batch[idx] = scale(img_batch[idx])

        tags_pred = model.net.predict(img_batch)
        tags_pred = tags_pred.round().astype(np.uint8)

        return tags_pred

if __name__ == "__main__":  # copied unmodified from vggnet.py and indiconv.py (except for the word ResNet50)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('ResNet50')

    parser = argparse.ArgumentParser(description='ResNet50 Model.')
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    parser_train = sub.add_parser('train', help='training')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-c', '--config', help='config file')
    parser_train.add_argument('-w', '--weights', help='network weights')

    # Prediction / submission.
    parser_predict = sub.add_parser('predict', help='make predictions')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('dataset', help='dataset', choices=['train', 'test'])
    parser_predict.add_argument('-c', '--config', help='config file')
    parser_predict.add_argument('-w', '--weights', help='network weights', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict']

    model = ResNet50()
    model.create_net()

    if args['weights'] is not None:
        logger.info('Loading network weights from %s.' % args['weights'])
        model.net.load_weights(args['weights'])

    if args['which'] == 'train':
        model.train()

    elif args['which'] == 'predict' and args['dataset'] == 'train':
        df = pd.read_csv('data/train_v2.csv')
        img_batch = np.empty([model.config['batch_size'], ] + model.config['input_shape'])
        F2_scores = []

        # Reading images, making predictions in batches.
        for idx in range(0, df.shape[0], model.config['batch_size']):

            # Read images, extract tags.
            img_names = df[idx:idx + model.config['batch_size']]['image_name'].values
            tags_true = df[idx:idx + model.config['batch_size']]['tags'].values
            for _, img_name in enumerate(img_names):
                try:
                    img_batch[_] = resize(tif.imread('data/train-tif-v2/%s.tif' % img_name),
                                          model.config['input_shape'][:2], preserve_range=True, mode='constant')
                except Exception as e:
                    logger.error('Bad image: %s' % img_name)
                    pass

            # Make predictions, compute F2 and store it.
            tags_pred = model.predict(img_batch)
            for tt, tp in zip(tags_true, tags_pred):
                tt = tagset_to_onehot(set(tt.split(' ')))
                F2_scores.append(onehot_F2(tt, tp))

            # Progress...
            logger.info('%d/%d, %.2lf, %.2lf' % (idx, df.shape[0], np.mean(F2_scores), np.mean(F2_scores[idx:])))

    elif args['which'] == 'predict' and args['dataset'] == 'test':
        df = pd.read_csv('data/sample_submission_v2.csv')
        img_batch = np.zeros([model.config['batch_size'], ] + model.config['input_shape'])
        submission_rows = []

        # Reading images, making predictions in batches.
        for idx in range(0, df.shape[0], model.config['batch_size']):

            # Read images.
            img_names = df[idx:idx + model.config['batch_size']]['image_name'].values
            for _, img_name in enumerate(img_names):
                try:
                    img_batch[_] = resize(tif.imread('data/test-tif-v2/%s.tif' % img_name),
                                          model.config['input_shape'][:2], preserve_range=True, mode='constant')
                except Exception as e:
                    logger.error('Bad image: %s' % img_name)
                    pass

            # Make predictions, store image name and tags as list of lists.
            tags_pred = model.predict(img_batch)
            for img_name, tp in zip(img_names, tags_pred):
                tp = ' '.join(onehot_to_taglist(tp))
                submission_rows.append([img_name, tp])

            # Progress...
            logger.info('%d/%d' % (idx, df.shape[0]))

        # Convert list of lists to dataframe and save.
        sub_path = '%s/submission_%s.csv' % (model.cpdir, str(int(time())))
        df_sub = pd.DataFrame(submission_rows, columns=['image_name', 'tags'])
        df_sub.to_csv(sub_path, index=False)
        logger.info('Submission saved to %s.' % sub_path)
