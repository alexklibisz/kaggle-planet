# Generic function that abstracts away all the model yuckiness.
import argparse
import logging
import tifffile as tif
import pandas as pd
import numpy as np
from time import time
from skimage.transform import resize
from scipy.misc import imread
from planet.utils.data_utils import bool_F2, tagset_to_ints, boolarray_to_taglist


def model_runner(model):

    assert hasattr(model, 'create_net') and callable(model.create_net)
    assert hasattr(model, 'predict') and callable(model.predict)

    assert 'batch_size_trn' in model.config
    assert 'batch_size_tst' in model.config
    assert 'input_shape' in model.config
    assert 'trn_imgs_dir' in model.config
    assert 'trn_imgs_csv' in model.config
    assert 'tst_imgs_dir' in model.config
    assert 'tst_imgs_csv' in model.config

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

    # Prediction / submission.
    parser_predict = sub.add_parser('predict', help='make predictions')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('dataset', help='dataset', choices=['train', 'test'])
    parser_predict.add_argument('-w', '--weights', help='network weights', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict', 'optimize']

    # Create network before loading weights or training.
    model.create_net(args['weights'])

    if args['which'] == 'train':
        model.train()

    elif args['which'] == 'predict' and args['dataset'] == 'train':
        df = pd.read_csv(model.config['trn_imgs_csv'])
        F2_scores = []

        # Reading images, making predictions in batches.
        for idx in range(0, df.shape[0], model.config['batch_size_tst']):

            # Read images, extract tags.
            img_names = df[idx:idx + model.config['batch_size_tst']]['image_name'].values
            tags_true = df[idx:idx + model.config['batch_size_tst']]['tags'].values
            tags_pred = model.predict(img_names)
            for tt, tp in zip(tags_true, tags_pred):
                tt = tagset_to_ints(set(tt.split(' ')))
                F2_scores.append(bool_F2(tt, tp))

            # Progress...
            logger.info('%d/%d F2 running = %.2lf, F2 batch = %.2lf' %
                        (idx, df.shape[0], np.mean(F2_scores), np.mean(F2_scores[idx:])))

        logger.info('F2 final = %lf' % np.mean(F2_scores))

    # Experimental threshold optimization.
    elif args['which'] == 'optimize':
        model.optimize()

    elif args['which'] == 'predict' and args['dataset'] == 'test':
        df = pd.read_csv(model.config['tst_imgs_csv'])
        submission_rows = []
        sub_path = '%s/submission_%s.csv' % (model.cpdir, str(int(time())))

        # Reading images, making predictions in batches.
        for idx in range(0, df.shape[0], model.config['batch_size_tst']):

            img_names = df[idx:idx + model.config['batch_size_tst']]['image_name'].values
            tags_pred = model.predict(img_names)

            for img_name, tp in zip(img_names, tags_pred):
                tp = ' '.join(boolarray_to_taglist(tp))
                submission_rows.append([img_name, tp])

            # Progress...
            logger.info('%d/%d' % (idx, df.shape[0]))

        # Convert list of lists to dataframe and save.
        df_sub = pd.DataFrame(submission_rows, columns=['image_name', 'tags'])
        df_sub.to_csv(sub_path, index=False)
        logger.info('Submission saved to %s.' % sub_path)
