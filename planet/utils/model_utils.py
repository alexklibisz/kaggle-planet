from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Reshape, concatenate, Lambda, BatchNormalization
from skimage.transform import resize
import tifffile as tif
import logging
import numpy as np
import keras.backend as K
import pandas as pd

from os import path
from planet.utils.data_utils import tagset_to_onehot, tagset_to_boolarray, onehot_to_taglist, TAGS, onehot_F2, random_transforms
from planet.utils.runtime import funcname

def train_batch_gen(model, csv_path='data/train.csv', imgs_dir='data/train-tif'):

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

    # # Build an index of tags to their corresponding indexes in the dataset
    # # so that you can sample tags evenly.
    # TAGS_cycle = cycle(TAGS)
    # tags_to_row_idxs = {t: [] for t in TAGS}
    # for idx, row in df.iterrows():
    #     for t in row['tags'].split(' '):
    #         tags_to_row_idxs[t].append(idx)

    while True:

        # New batches at each iteration to prevent over-writing previous batch before it's used.
        imgs_batch = np.zeros([model.config['batch_size'], ] + model.config['input_shape'], dtype=np.float32)
        tags_batch = np.zeros([model.config['batch_size'], ] + model.config['output_shape'], dtype=np.uint8)

        # Sample *model.config['batch_size']* random rows and build the batches.
        for batch_idx in range(model.config['batch_size']):
            # data_idx = model.rng.choice(tags_to_row_idxs[next(TAGS_cycle)])
            data_idx = model.rng.randint(0, len(img_names))

            # try:
            img = resize(tif.imread(img_names[data_idx]), model.config['input_shape'][:2], preserve_range=True, mode='constant')
            if model.config['trn_transform']:
                imgs_batch[batch_idx] = scale(random_transforms(img, nb_min=0, nb_max=5))
            else:
                imgs_batch[batch_idx] = scale(img)
            tags_batch[batch_idx] = model.out.tagset_to_output(tag_sets[data_idx])

            # except Exception as e:
            #     print('Exception!')
            #     pass

        yield imgs_batch, tags_batch
