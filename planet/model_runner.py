import argparse
import h5py
import json
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from scipy.misc import imresize
from time import time
from tqdm import tqdm
import sys
sys.path.append('.')
from planet.utils.data_utils import optimize_thresholds, tags_f2pr, f2pr, TAGS, binary_to_tagstr
from planet.utils.runtime import funcname

np.random.seed(int(time()) + os.getpid())
tf.set_random_seed(1 + int(time()) + os.getpid())

# Overwrite the random seed
np.random.seed(317)
tf.set_random_seed(318)


def train(model_class, args):

    model = model_class(model_path=args['model'])
    model.serialize()
    return model.train()


def submission(names, yp, csv_path):
    logger = logging.getLogger(funcname())
    df_rows = [[names[i], binary_to_tagstr(yp[i, :])] for i in range(yp.shape[0])]
    df_sub = pd.DataFrame(df_rows, columns=['image_name', 'tags'])
    df_sub.to_csv(csv_path, index=False)
    logger.info('Saved %s.' % csv_path)


def predict(model_class, args):
    """Instantiates the model and makes augmented predictions. Serializes the predictions
    as numpy matrices."""

    logger = logging.getLogger(funcname())

    # Setup model.
    model = model_class(model_path=args['model'])
    model.cfg['cpdir'] = '/'.join(args['model'].split('/')[:-1])
    assert 'hdf5_path_trn' in model.cfg
    assert 'hdf5_path_tst' in model.cfg
    assert 'tst_batch_size' in model.cfg

    # Load data.
    data_trn = h5py.File(model.cfg['hdf5_path_trn'], 'r')
    data_tst = h5py.File(model.cfg['hdf5_path_tst'], 'r')
    imgs_trn, tags_trn = data_trn.get('images'), data_trn.get('tags')
    imgs_tst, tags_tst = data_tst.get('images'), data_tst.get('tags')

    # Make training and testing predictions batch-by-batch for multiple
    # augmentations. Serialize the complete matrix of activations for each
    # augmentation separately.
    aug_funcs = [
        ('identity', lambda x: x),
        ('vflip', lambda x: x[:, ::-1, ...]),
        ('hflip', lambda x: x[:, :, ::-1]),
        ('rot90', lambda x: np.rot90(x, 1, axes=(1, 2))),
        ('rot180', lambda x: np.rot90(x, 2, axes=(1, 2))),
        ('rot270', lambda x: np.rot90(x, 3, axes=(1, 2))),
        ('rot90vflip', lambda x: np.rot90(x, 1, axes=(1, 2))[:, ::-1, ...]),
        ('rot90hflip', lambda x: np.rot90(x, 1, axes=(1, 2))[:, :, ::-1])
    ]

    for aug_name, aug_func in aug_funcs:

        # Train set.
        logger.info('TTA %s.' % (aug_name))
        yt = tags_trn[...]
        yp = np.zeros(tags_trn.shape, dtype=np.float32)

        for i0 in tqdm(range(0, imgs_trn.shape[0], model.cfg['tst_batch_size'])):
            i1 = i0 + min(model.cfg['tst_batch_size'], imgs_trn.shape[0] - i0)
            ib = np.array([imresize(img[...], model.cfg['input_shape']) for img in imgs_trn[i0:i1]])
            yp[i0:i1] = model.predict_batch(aug_func(ib))

        # Optimize activation thresholds and print F2 as a sanity check.
        f2, p, r = f2pr(yt, (yp > 0.5).astype(np.uint8))
        logger.info('Default   f2=%.4lf, p=%.4lf, r=%.4lf' % (f2, p, r))

        thresh_opt = optimize_thresholds(yt, yp)
        f2, p, r = f2pr(yt, (yp > thresh_opt).astype(np.uint8))
        logger.info('Optimized f2=%.4lf, p=%.4lf, r=%.4lf' % (f2, p, r))

        # Save raw activations.
        npy_path = '%s/yp_trn_%s.npy' % (model.cpdir, aug_name)
        np.save(npy_path, yp)
        logger.info('Saved %s.' % npy_path)

        # Test set.
        yp = np.zeros(tags_tst.shape, dtype=np.float32)
        ib = np.zeros((model.cfg['tst_batch_size'], *model.net.input_shape[1:]))
        for i0 in tqdm(range(0, imgs_tst.shape[0], model.cfg['tst_batch_size'])):
            i1 = i0 + min(model.cfg['tst_batch_size'], imgs_tst.shape[0] - i0)
            for i in range(i1 - i0):
                ib[i] = imresize(imgs_tst[i, :, :, :], model.net.input_shape[1:])
            yp[i0:i1] = model.predict_batch(aug_func(ib[:i1 - i0]))

        # Save raw activations
        npy_path = '%s/yp_tst_%s.npy' % (model.cpdir, aug_name)
        np.save(npy_path, yp)
        logger.info('Saved %s.' % npy_path)


def model_runner(model_class):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Model runner.')
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    p = sub.add_parser('train', help='training')
    p.set_defaults(which='train')
    p.add_argument('-j', '--model', help='path to serialized model')

    # Prediction.
    p = sub.add_parser('predict', help='make predictions')
    p.set_defaults(which='predict')
    p.add_argument('-j', '--model', help='path to serialized model', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict']

    if args['which'] == 'train':
        return train(model_class, args)

    if args['which'] == 'predict':
        return predict(model_class, args)
