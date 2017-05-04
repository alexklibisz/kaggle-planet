from glob import glob
from keras.utils.np_utils import to_categorical
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


def random_transforms(img, nb_min=0, nb_max=5, rng=np.random):

    transforms = [
        lambda x: np.rot90(x, k=rng.randint(1, 4), axes=(0, 1)),
        lambda x: np.flipud(x),
        lambda x: np.fliplr(x),
        lambda x: sktf.rotate(x, angle=rng.randint(1, 360), mode='reflect', preserve_range=True)
    ]

    nb = rng.randint(nb_min, nb_max)
    for _ in range(nb):
        img = rng.choice(transforms)(img)

    return img


# def get_imgs_lbls(imgs_dir, lbls_path):
#     nb_imgs = len(glob('%s/*.tif' % imgs_dir))
#     scale = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) * 2 - 1
#     name_to_path = lambda x: '%s/%s.tif' % (imgs_dir, x)
#     df = pd.read_csv(lbls_path)
#     imgs = np.zeros((nb_imgs, 256, 256, 4), dtype=np.float32)
#     lbls = np.zeros((nb_imgs, 17, 2), dtype=np.uint8)
#     idx = 0
#     for _, row in df.iterrows():
#         if not path.exists(name_to_path(row['image_name'])):
#             continue
#         imgs[idx] = scale(tif.imread(name_to_path(row['image_name'])))
#         row_lbls = row['tags'].split(' ')
#         for lbl in TAGS:
#             hot_idx = int(lbl in row_lbls)
#             lbl_idx = TAG_TO_IDX[lbl]
#             lbls[idx][lbl_idx][hot_idx] = 1
#         idx += 1

#     return imgs, lbls
