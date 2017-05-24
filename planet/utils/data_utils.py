from os import path, listdir
from sklearn.metrics import precision_score, recall_score, f1_score
import tifffile as tif
import numpy as np
import pandas as pd
import skimage.transform as sktf
from planet.utils.runtime import memory_usage


# Never change the order here.
TAGS = sorted(['agriculture', 'artisinal_mine', 'bare_ground', 'blooming', 'blow_down', 'clear', 'cloudy', 'conventional_mine',
               'cultivation', 'habitation', 'haze', 'partly_cloudy', 'primary', 'road', 'selective_logging', 'slash_burn', 'water'])
# TAG_TO_IDX = {l: idx for idx, l in enumerate(TAGS)}


def tagset_to_onehot(tagset):
    tags = np.zeros((len(TAGS), 2), dtype=np.uint8)
    for idx, tag in enumerate(TAGS):
        tags[idx][int(tag in tagset)] = 1
    return tags

def tagset_to_boolarray(tagset):
    tags = np.zeros((len(TAGS), 1), dtype=np.uint8)
    for idx, tag in enumerate(TAGS):
        if tag in tagset:
            tags[idx] = 1
    return tags

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


def random_transforms(img, nb_min=0, nb_max=2, rng=np.random):

    transforms = [
        lambda x: x,
        lambda x: np.rot90(x, k=rng.randint(1, 4), axes=(0, 1)),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: np.roll(x, rng.randint(1, x.shape[0]), 0),
        lambda x: np.roll(x, rng.randint(1, x.shape[1]), 1),
        lambda x: sktf.rotate(x, angle=rng.randint(1, 360), preserve_range=True, mode='reflect'),

        # Resize up to 4px in horizontal or vertical direction. Crop starting at top left or bottom right.
	lambda x: sktf.resize(x, (x.shape[0] + rng.randint(0, 4), x.shape[0] + rng.randint(0, 4)),
                              preserve_range=True, mode='constant')[:x.shape[0], :x.shape[1], :],
        lambda x: sktf.resize(x, (x.shape[0] + rng.randint(0, 4), x.shape[0] + rng.randint(0, 4)),
                              preserve_range=True, mode='constant')[-x.shape[0]:, -x.shape[1]:, :],
    ]

    nb = rng.randint(nb_min, nb_max)
    for _ in range(nb):
        img = rng.choice(transforms)(img)

    return img
