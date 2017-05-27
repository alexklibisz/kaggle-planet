# Generic function that abstracts away all the model yuckiness.
import argparse
import logging
import tifffile as tif
import pandas as pd
import numpy as np
from skimage.transform import resize
from scipy.misc import imread 
from planet.utils.data_utils import bool_F2, tagset_to_boolarray, boolarray_to_taglist

def model_runner(model_class):
    logging.basicConfig(level=logging.INFO)
    model_name = model_class.__name__
    logger = logging.getLogger(model_name)

    parser = argparse.ArgumentParser(description='%s Model...' % model_name)
    sub = parser.add_subparsers(title='actions', description='Choose an action.')

    # Training.
    parser_train = sub.add_parser('train', help='training')
    parser_train.set_defaults(which='train')
    parser_train.add_argument('-w', '--weights', help='network weights')

    # Prediction / submission.
    parser_predict = sub.add_parser('predict', help='make predictions')
    parser_predict.set_defaults(which='predict')
    parser_predict.add_argument('dataset', help='dataset', choices=['train', 'test'])
    parser_predict.add_argument('-w', '--weights', help='network weights', required=True)

    args = vars(parser.parse_args())
    assert args['which'] in ['train', 'predict']

    # Create model
    model = model_class()
    assert 'batch_size' in model.config
    assert 'input_shape' in model.config
    assert 'imgs_dir' in model.config
    model.create_net()

    # Set up functions for reading images
    if 'tif' in model.config['imgs_dir']:
        read_training_image = lambda img_name: tif.imread('%s/%s.tif' % (model.config['imgs_dir'], img_name))
        read_testing_image  = lambda img_name: tif.imread('%s/%s.tif' % ('data/test-tif-v2', img_name))
    elif 'jpg' in model.config['imgs_dir']:
        read_training_image = lambda img_name: imread('%s/%s.jpg' % (model.config['imgs_dir'], img_name), mode='RGB')
        read_testing_image  = lambda img_name: imread('%s/%s.jpg' % ('data/test-jpg-v2', img_name), mode='RGB')
    else:
        logger.error('imgs_dir must have tif or jpg in it so I know how to read the images: %s' % model.config['imgs_dir'])
        return 1

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
                img_batch[_] = resize(read_training_image(img_name),
                                      model.config['input_shape'][:2], preserve_range=True, mode='constant')

            # Make predictions, compute F2 and store it.
            tags_pred = model.predict(img_batch)
            for tt, tp in zip(tags_true, tags_pred):
                tt = tagset_to_boolarray(set(tt.split(' ')))
                F2_scores.append(bool_F2(tt, tp))

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
                img_batch[_] = resize(read_testing_image(img_name),
                                      model.config['input_shape'][:2], preserve_range=True, mode='constant')

            # Make predictions, store image name and tags as list of lists.
            tags_pred = model.predict(img_batch)
            for img_name, tp in zip(img_names, tags_pred):
                tp = ' '.join(boolarray_to_taglist(tp))
                submission_rows.append([img_name, tp])

            # Progress...
            logger.info('%d/%d' % (idx, df.shape[0]))

        # Convert list of lists to dataframe and save.
        sub_path = '%s/submission_%s.csv' % (model.cpdir, str(int(time())))
        df_sub = pd.DataFrame(submission_rows, columns=['image_name', 'tags'])
        df_sub.to_csv(sub_path, index=False)
        logger.info('Submission saved to %s.' % sub_path)