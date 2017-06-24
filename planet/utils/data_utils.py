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

TAGS_short = [t[:4] for t in TAGS]


def correct_tags(tags):

    # "cloudy" nullifies all other predictions.
    cloudy_idx = 6
    if np.argmax(tags) == cloudy_idx:
        tags *= 0
        tags[cloudy_idx] = 1.

    # Pick the maximum cloud-cover and remove all others.
    cc_idxs = [5, 6, 10, 11]
    cc_msk = np.zeros(len(TAGS), dtype=np.uint8)
    cc_msk[cc_idxs] = 1.
    cc_max_idx = np.argmax(tags * cc_msk)
    tags[cc_idxs] = 0.
    tags[cc_max_idx] = 1.

    return tags


def tagstr_to_binary(tagstr):
    tagset = set(tagstr.strip().split(' '))
    tags = np.zeros((len(TAGS)), dtype=np.uint8)
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


def random_transforms(img, nb_min=0, nb_max=2, rng=np.random):

    transforms = [
        lambda x: x,
        lambda x: np.rot90(x, k=rng.randint(1, 4), axes=(0, 1)),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        # lambda x: np.roll(x, rng.randint(1, x.shape[0]), 0),
        # lambda x: np.roll(x, rng.randint(1, x.shape[1]), 1),
        # lambda x: sktf.rotate(x, angle=rng.randint(1, 360), preserve_range=True, mode='reflect'),

        # # Resize up to 4px in horizontal or vertical direction. Crop starting at top left or bottom right.
        # lambda x: sktf.resize(x, (x.shape[0] + rng.randint(0, 4), x.shape[1] + rng.randint(0, 4)),
        #                       preserve_range=True, mode='constant')[:x.shape[0], :x.shape[1], :],
        # lambda x: sktf.resize(x, (x.shape[0] + rng.randint(0, 4), x.shape[1] + rng.randint(0, 4)),
        #                       preserve_range=True, mode='constant')[-x.shape[0]:, -x.shape[1]:, :],
    ]

    nb = rng.randint(nb_min, nb_max)
    for _ in range(nb):
        img = rng.choice(transforms)(img)

    return img
