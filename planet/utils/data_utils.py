from os import path, listdir
from sklearn.metrics import precision_score, recall_score, f1_score
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


def onehot_to_taglist(onehot):
    taglist = []
    for idx, tag in enumerate(TAGS):
        if onehot[idx][1] == 1:
            taglist.append(tag)
    return taglist


def onehot_precision(A, B):
    return precision_score(A[:, 1], B[:, 1])


def onehot_recall(A, B):
    return recall_score(A[:, 1], B[:, 1])


def onehot_F2(A, B):
    beta = 2
    p = onehot_precision(A, B)
    r = onehot_recall(A, B)
    return (1 + beta**2) * ((p * r) / (beta**2 * p + r + 1e-7))


def bool_F2(A, B):
    beta = 2
    p = precision_score(A, B)
    r = recall_score(A, B)
    return (1 + beta**2) * ((p * r) / (beta**2 * p + r + 1e-7))


def val_plot_metrics(json_path):

    return


def val_plot_predictions(json_path):

    return
