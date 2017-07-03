# Converts dataset (imgs + csv file with tags) into HDF5 format containing its images and tags.
# Also prints out the mean for each channel.
from glob import glob
from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np
import pandas as pd

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, tagstr_to_binary


def hdf5_jpgs(dir_jpg, path_csv, path_hdf5):
    '''Make hdf5 file with 'images/[index]' datasets for each image and a single 'tags' dataset
    containing the array of binary tag arrays.'''
    df = pd.read_csv(path_csv)
    f = h5py.File(path_hdf5, 'w')
    mean_img = np.zeros((256, 256, 3))

    for i, row in tqdm(df.iterrows()):
        name = row['image_name']
        jpg = Image.open('%s/%s.jpg' % (dir_jpg, name)).convert('RGB')
        ds = f.create_dataset('images/%d' % i, (256, 256, 3), dtype='uint8', chunks=(256, 256, 3))
        ds[...] = np.asarray(jpg)
        mean_img += ds[...] / df.shape[0]
        assert np.all(ds[...] == jpg)

    print('Channel means:')
    print('Ch 0', np.mean(mean_img[:, :, 0]))
    print('Ch 1', np.mean(mean_img[:, :, 1]))
    print('Ch 2', np.mean(mean_img[:, :, 2]))

    tags = np.zeros((df.shape[0], len(TAGS)), dtype=np.int8)
    for i, row in tqdm(df.iterrows()):
        tagstr = row['tags']
        tags[i] = tagstr_to_binary(tagstr)

    ds = f.create_dataset('tags', tags.shape, dtype='int8')
    ds[...] = tags

    f.close()

    # Test the read speed.
    f = h5py.File(path_hdf5, 'r')
    dsnames = ['images/%d' % i for i in range(df.shape[0])]
    for i in tqdm(range(df.shape[0])):
        img = f.get(dsnames[i])[...]


hdf5_jpgs('data/train-jpg', 'data/train_v2.csv', 'data/train-jpg.hdf5')
hdf5_jpgs('data/test-jpg', 'data/sample_submission_v2.csv', 'data/test-jpg.hdf5')
