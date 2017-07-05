# Converts dataset (imgs + csv file with tags) into HDF5 format containing its images and tags.
# Also prints out the mean for each channel.
# With a ~3 year old SSD, jpgs read at ~3000/s and tifs at ~400/s.
from glob import glob
from PIL import Image
from tqdm import tqdm
import h5py
import numpy as np
import os
import pandas as pd
import tifffile as tif

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, tagstr_to_binary


def read_jpg(path):
    path += '.jpg'
    img = Image.open(path).convert('RGB')
    return np.asarray(img)


def read_tif(path):
    path += '.tif'
    return tif.imread(path)


def make_hdf5(imgs_dir, img_shape, img_dtype, path_csv, path_hdf5):
    '''Make hdf5 file with 'images/[index]' datasets for each image and a single 'tags' dataset
    containing the array of binary tag arrays.'''
    df = pd.read_csv(path_csv)
    f = h5py.File(path_hdf5, 'w')
    mean_img = np.zeros(img_shape)

    read_func = read_jpg if img_shape[-1] == 3 else read_tif

    for i, row in tqdm(df.iterrows()):
        name = row['image_name']
        ds = f.create_dataset('images/%d' % i, img_shape, dtype=img_dtype, chunks=img_shape)
        img = read_func('%s/%s' % (imgs_dir, name))
        ds[...] = img
        mean_img += ds[...] / df.shape[0]
        assert np.all(ds[...] == img)

    print('Channel means:')
    for c in range(img_shape[-1]):
        print('Ch %d: %.8lf' % (c, np.mean(mean_img[:, :, c])))

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


if not os.path.exists('data/train-jpg.hdf5'):
    make_hdf5('data/train-jpg', (256, 256, 3), 'uint8', 'data/train_v2.csv', 'data/train-jpg.hdf5')

if not os.path.exists('data/test-jpg.hdf5'):
    make_hdf5('data/test-jpg', (256, 256, 3), 'uint8', 'data/sample_submission_v2.csv', 'data/test-jpg.hdf5')

if not os.path.exists('data/train-tif.hdf5'):
    make_hdf5('data/train-tif-v2', (256, 256, 4), 'uint16', 'data/train_v2.csv', 'data/train-tif.hdf5')

if not os.path.exists('data/test-tif.hdf5'):
    make_hdf5('data/test-tif-v2', (256, 256, 4), 'uint16', 'data/sample_submission_v2.csv', 'data/test-tif.hdf5')
