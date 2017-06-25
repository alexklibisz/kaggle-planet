# Generic function that abstracts away all the model yuckiness.
import argparse
import logging
import tifffile as tif
import pandas as pd
import numpy as np
from time import time
from skimage.transform import resize
from scipy.misc import imread
from sklearn.metrics import precision_score, recall_score
from planet.utils.data_utils import bool_F2, tagstr_to_binary, binary_to_tagstr, correct_tags


def model_runner(model):

    assert hasattr(model, 'create_net') and callable(model.create_net)
    assert hasattr(model, 'predict') and callable(model.predict)

    assert 'batch_size_trn' in model.config
    assert 'batch_size_tst' in model.config
    assert 'input_shape' in model.config
    assert 'imgs_dir_trn' in model.config
    assert 'imgs_csv_trn' in model.config
    assert 'imgs_dir_tst' in model.config
    assert 'imgs_csv_tst' in model.config

    logging.basicConfig(level=logging.INFO)
    model_name = type(model).__name__
    logger = logging.getLogger(model_name)

    parser = argparse.ArgumentParser(description='%s Model...' % model_name)
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    parser_train = sub.add_parser('train', help='training')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-w', '--weights', help='network weights')

    # Optimizing.
    parser_train = sub.add_parser('optimize', help='optimizing')
    parser_train.set_defaults(which='optimize')
    parser_train.add_argument('-w', '--weights', help='network weights')

    # Optimizing.
    parser_train = sub.add_parser('visualize', help='optimizing')
    parser_train.set_defaults(which='visualize')
    parser_train.add_argument('-w', '--weights', help='network weights')

    # Prediction / submission.
    parser_predict = sub.add_parser('predict', help='make predictions')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('dataset', help='dataset', choices=['train', 'test'])
    parser_predict.add_argument('-w', '--weights', help='network weights', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict', 'optimize', 'visualize']

    # Create network before loading weights or training.
    model.create_net(args['weights'])

    if args['which'] == 'train':
        model.train()

    elif args['which'] == 'predict' and args['dataset'] == 'train':
        df = pd.read_csv(model.config['imgs_csv_trn'])
        prec_scores, reca_scores, F2_scores = [], [], []

        # Reading images, making predictions in batches, tracking F2 scores.
        for idx in range(0, df.shape[0], model.config['batch_size_tst']):

            # Read images, extract tags.
            img_names = df[idx:idx + model.config['batch_size_tst']]['image_name'].values
            tags_true = df[idx:idx + model.config['batch_size_tst']]['tags'].values
            tags_pred = model.predict(img_names)
            for tt, tp in zip(tags_true, tags_pred):
                tt = tagstr_to_binary(tt)
                F2_scores.append(bool_F2(tt, tp))
                prec_scores.append(precision_score(tt, tp))
                reca_scores.append(recall_score(tt, tp))

            # Progress and metrics...
            logger.info('%d/%d F2 running=%.4lf, F2 batch=%.4lf' %
                        (idx, df.shape[0], np.mean(F2_scores), np.mean(F2_scores[idx:])))
            logger.info('%d/%d prec running=%.4lf, prec batch=%.4lf' %
                        (idx, df.shape[0], np.mean(prec_scores), np.mean(prec_scores[idx:])))
            logger.info('%d/%d reca running=%.4lf, reca batch=%.4lf' %
                        (idx, df.shape[0], np.mean(reca_scores), np.mean(reca_scores[idx:])))

        logger.info('F2 final = %lf' % np.mean(F2_scores))

    elif args['which'] == 'predict' and args['dataset'] == 'test':
        df = pd.read_csv(model.config['imgs_csv_tst'])
        submission_rows = []
        sub_path = '%s/submission_%s.csv' % (model.cpdir, str(int(time())))

        # Reading images, making predictions in batches.
        for idx in range(0, df.shape[0], model.config['batch_size_tst']):

            img_names = df[idx:idx + model.config['batch_size_tst']]['image_name'].values
            tags_pred = model.predict(img_names)

            for img_name, tp in zip(img_names, tags_pred):
                tpstr = binary_to_tagstr(tp)
                submission_rows.append([img_name, tpstr])

            # Progress...
            logger.info('%d/%d' % (idx, df.shape[0]))

        # Convert list of lists to dataframe and save.
        df_sub = pd.DataFrame(submission_rows, columns=['image_name', 'tags'])
        df_sub.to_csv(sub_path, index=False)
        logger.info('Submission saved to %s.' % sub_path)

    # Experimental threshold optimization.
    elif args['which'] == 'optimize':
        model.optimize()

    elif args['which'] == 'visualize':
        model.visualize_activation()
