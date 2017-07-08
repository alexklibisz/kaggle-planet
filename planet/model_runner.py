import argparse
import h5py
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
import sys
sys.path.append('.')
from planet.utils.data_utils import optimize_thresholds, tags_f2pr, f2pr, TAGS, binary_to_tagstr
from planet.utils.runtime import funcname

np.random.seed(317)
tf.set_random_seed(318)


def train(model_class, args):

    if args['json']:
        fp = open(args['json'], 'r')
        args['json'] = json.load(fp)
        fp.close()

    model = model_class(model_json=args['json'], weights_path=args['weights'])
    model.serialize()
    return model.train()


def submission(names, yp, csv_path):
    logger = logging.getLogger(funcname())
    df_rows = [[names[i], binary_to_tagstr(yp[i, :])] for i in range(yp.shape[0])]
    df_sub = pd.DataFrame(df_rows, columns=['image_name', 'tags'])
    df_sub.to_csv(csv_path, index=False)
    logger.info('Saved %s.' % csv_path)


def predict(model_class, args):
    """Uses the given model to make predictions, convert them into strings, and save
    them to a CSV file. Uses the training data to pick the best threshold for each 
    output activation. Saves both the optimized and regular thresholds."""

    logger = logging.getLogger(funcname())

    if args['json']:
        fp = open(args['json'], 'r')
        args['json'] = json.load(fp)
        fp.close()

    # Predictions for training data.
    model = model_class(args['json'], args['weights'])
    model.cfg['cpdir'] = '/'.join(args['weights'].split('/')[:-1])
    names, yt_trn, yp_trn = model.predict('train')

    # Compute thresholds for each tag.
    logger.info('Optimizing thresholds')
    thresh_def = np.ones(len(TAGS)) * 0.5
    thresh_opt = optimize_thresholds(yt_trn, yp_trn)
    logger.info('Optimized thresholds: %s' % str(thresh_opt))

    yp_trn_def = (yp_trn > thresh_def).astype(np.uint8)
    yp_trn_opt = (yp_trn > thresh_opt).astype(np.uint8)
    f2t_def, pt_def, rt_def = tags_f2pr(yt_trn, yp_trn_def, thresh_def)
    f2t_opt, pt_opt, rt_opt = tags_f2pr(yt_trn, yp_trn_opt, thresh_opt)

    # Print metrics for each tag.
    for i, t in enumerate(TAGS):
        logger.info('%-20s f2=%.3lf p=%.3lf r=%.3lf t=%.3lf' % (t, f2t_def[i], pt_def[i], rt_def[i], thresh_def[i]))
        logger.info('%-20s f2=%.3lf p=%.3lf r=%.3lf t=%.3lf' % (t, f2t_opt[i], pt_opt[i], rt_opt[i], thresh_opt[i]))

    # Print aggregate metrics.
    f2_def, p_def, r_def = f2pr(yt_trn, yp_trn_def)
    f2_opt, p_opt, r_opt = f2pr(yt_trn, yp_trn_opt)
    logger.info('%-20s f2=%.3lf p=%.3lf r=%.3lf' % ('Def. aggregate', f2_def, p_def, r_def))
    logger.info('%-20s f2=%.3lf p=%.3lf r=%.3lf' % ('Opt. aggregate', f2_opt, p_opt, r_opt))

    # Save submissions for training and testing with and without optimization.
    t = int(time())
    csv_path = '%s/submission_trn_def_%d.csv' % (model.cpdir, t)
    submission(names, yp_trn_def, csv_path)

    csv_path = '%s/submission_trn_opt_%d.csv' % (model.cpdir, t)
    submission(names, yp_trn_opt, csv_path)

    names, _, yp_tst = model.predict('test')
    yp_tst_def = (yp_tst > thresh_def).astype(np.uint8)
    csv_path = '%s/submission_tst_def_%d.csv' % (model.cpdir, t)
    submission(names, yp_tst_def, csv_path)

    yp_tst_opt = (yp_tst > thresh_opt).astype(np.uint8)
    csv_path = '%s/submission_tst_opt_%d.csv' % (model.cpdir, t)
    submission(names, yp_tst_opt, csv_path)

    # Save raw activations.
    np_path = '%s/yp_trn_%d.npy' % (model.cpdir, t)
    np.save(np_path, yp_trn)
    logger.info('Saved %s.' % np_path)

    np_path = '%s/yp_tst_%d.npy' % (model.cpdir, t)
    np.save(np_path, yp_tst)
    logger.info('Saved %s.' % np_path)


def model_runner(model_class):

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='Model runner.')
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    p = sub.add_parser('train', help='training')
    p.set_defaults(which='train')
    p.add_argument('-j', '--json', help='model JSON definition')
    p.add_argument('-w', '--weights', help='network weights')

    # Prediction.
    p = sub.add_parser('predict', help='make predictions')
    p.set_defaults(which='predict')
    p.add_argument('-j', '--json', help='model JSON definition')
    p.add_argument('-w', '--weights', help='path to network weights', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict']

    if args['which'] == 'train':
        return train(model_class, args)

    if args['which'] == 'predict':
        return predict(model_class, args)
