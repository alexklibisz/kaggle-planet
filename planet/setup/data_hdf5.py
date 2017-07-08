# Converts dataset (imgs + csv file with tags) into HDF5 format containing its images and tags.
# Also prints out the mean for each channel.
# With a ~3 year old SSD, jpgs read at ~3000/s and tifs at ~400/s.
# Structure of the dataset hdf5 file is as follows:
# 'images' is a (no. images, height, width, channels) array.
# 'tags' is a (no. images, 17) array.
# attrs['names'] contains the ordered list of file names (test_0, test_1, ...) as a comma-separated string.

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
    names, tagstrs = df['image_name'].values, df['tags'].values
    f = h5py.File(path_hdf5, 'w')
    mean_img = np.zeros(img_shape)
    read_func = read_jpg if img_shape[-1] == 3 else read_tif

    # Set the image names as an attribute.
    f.attrs['names'] = ','.join(df['image_name'].values)

    # Single dataset for all images.
    ds = f.create_dataset('images', (df.shape[0], *img_shape), dtype=img_dtype, chunks=(1, *img_shape))

    for i, name in tqdm(enumerate(names)):
        img = read_func('%s/%s' % (imgs_dir, name))
        ds[i, :, :, :] = img
        mean_img += ds[i, :, :, :] / df.shape[0]
        assert np.all(ds[i, :, :, :] == img)

    print('Channel means:')
    for c in range(img_shape[-1]):
        print('Ch %d: %.8lf' % (c, np.mean(mean_img[:, :, c])))

    tags = np.zeros((df.shape[0], len(TAGS)), dtype=np.int8)
    for i, tagstr in enumerate(tagstrs):
        tags[i] = tagstr_to_binary(tagstr)

    ds = f.create_dataset('tags', tags.shape, dtype='int8')
    ds[...] = tags
    f.close()

    # Test the read speed.
    f = h5py.File(path_hdf5, 'r')
    imgs = f.get('images')
    for i, name in tqdm(enumerate(names)):
        img = imgs[i, :, :, :]


if not os.path.exists('data/train-jpg.hdf5'):
    make_hdf5('data/train-jpg', (256, 256, 3), 'uint8', 'data/train_v2.csv', 'data/train-jpg.hdf5')

if not os.path.exists('data/test-jpg.hdf5'):
    make_hdf5('data/test-jpg', (256, 256, 3), 'uint8', 'data/sample_submission_v2.csv', 'data/test-jpg.hdf5')

if not os.path.exists('data/train-tif.hdf5'):
    make_hdf5('data/train-tif-v2', (256, 256, 4), 'uint16', 'data/train_v2.csv', 'data/train-tif.hdf5')

if not os.path.exists('data/test-tif.hdf5'):
    make_hdf5('data/test-tif-v2', (256, 256, 4), 'uint16', 'data/sample_submission_v2.csv', 'data/test-tif.hdf5')
