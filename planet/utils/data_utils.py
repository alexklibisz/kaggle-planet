from os import path, listdir
from scipy.stats import entropy
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import h5py
import tifffile as tif
import numpy as np
import pandas as pd
import skimage.transform as sktf


# Never change the order here.
TAGS = [
    'agriculture',              # 0
    'artisinal_mine',           # 1
    'bare_ground',              # 2
    'blooming',                 # 3
    'blow_down',                # 4
    'clear',                    # 5
    'cloudy',                   # 6
    'conventional_mine',        # 7
    'cultivation',              # 8
    'habitation',               # 9
    'haze',                     # 10
    'partly_cloudy',            # 11
    'primary',                  # 12
    'road',                     # 13
    'selective_logging',        # 14
    'slash_burn',               # 15
    'water']                    # 16

# Abbreviations of tags.
TAGS_short = [t[:4] for t in TAGS]

# Lookup tag's index by its string value.
TAGS_idxs = {t: i for i, t in enumerate(TAGS)}

# Mean values per channel computed on train dataset.
IMG_MEAN_JPG_TRN = (123.7213287353515625, 124.1055145263671875, 107.1297149658203125)
IMG_MEAN_TIF_TRN = (5023.3525390625, 4288.53466796875, 2961.06689453125, 6369.85498046875)


def tag_proportions(csvpath='data/train_v2.csv'):
    df = pd.read_csv(csvpath)
    t = np.vstack([tagstr_to_binary(ts) for ts in df['tags'].values])
    ppos = np.sum(t, axis=0) / t.shape[0]
    pneg = np.sum(np.abs(t - 1), axis=0) / t.shape[0]
    return ppos, pneg


def correct_tags(tags):

    # Pick the maximum cloud-cover and remove all others.
    cc_idxs = [
        TAGS_idxs['clear'],
        TAGS_idxs['cloudy'],
        TAGS_idxs['haze'],
        TAGS_idxs['partly_cloudy']
    ]
    cc_msk = np.zeros(len(TAGS), dtype=np.int8)
    cc_msk[cc_idxs] = 1.
    cc_max_idx = np.argmax(tags * cc_msk)
    tags[cc_idxs] = 0.
    tags[cc_max_idx] = 1.
    assert np.sum(tags[cc_idxs]) == 1.

    # "cloudy" nullifies all other predictions.
    cloudy_idx = TAGS_idxs['cloudy']
    if cloudy_idx == cc_max_idx:
        tags *= 0
        tags[cloudy_idx] = 1.
        assert np.sum(tags) == 1.

    return tags


def get_train_val_idxs(hdf5_path, prop_trn=0.8, rng=None, nb_iter=1000):
    '''Picks the random training and validation indexes from the given array of tags
    that minimizes the mean absolute error relative the full dataset.'''
    if rng is None:
        rng = np.random

    f = h5py.File(hdf5_path, 'r')
    tags_binary = f.get('tags')[...].astype(np.float32)
    f.close()

    dist_full = np.sum(tags_binary, axis=0) / len(tags_binary)
    best, min_mae = None, 1e10
    idxs = np.arange(tags_binary.shape[0])
    splt = int(tags_binary.shape[0] * prop_trn)
    for _ in range(nb_iter):
        rng.shuffle(idxs)
        idxs_trn, idxs_val = idxs[:splt], idxs[splt:]
        assert set(idxs_trn).intersection(idxs_val) == set([])
        dist_trn = np.sum(tags_binary[idxs_trn], axis=0) / len(idxs_trn)
        dist_val = np.sum(tags_binary[idxs_val], axis=0) / len(idxs_val)
        if np.count_nonzero(dist_val) < dist_val.shape[0]:
            continue
        mae = np.mean(np.abs(dist_full - dist_trn)) + np.mean(np.abs(dist_full - dist_val))
        if mae < min_mae:
            min_mae = mae
            best = (idxs_trn, idxs_val)
    return best


def tagstr_to_binary(tagstr):
    tagset = set(tagstr.strip().split(' '))
    tags = np.zeros((len(TAGS)), dtype=np.int8)
    for idx, tag in enumerate(TAGS):
        if tag in tagset:
            tags[idx] = 1
    return tags


def binary_to_tagstr(binary):
    s = ''
    for idx, tag in enumerate(TAGS):
        if binary[idx] == 1:
            s += tag + ' '
    return s.strip()


def bool_F2(A, B):
    beta = 2
    p = precision_score(A, B)
    r = recall_score(A, B)
    return (1 + beta**2) * ((p * r) / (beta**2 * p + r + 1e-7))


def val_plot_metrics(pickle_path):

    return


def val_plot_predictions(pickle_path):

    return
