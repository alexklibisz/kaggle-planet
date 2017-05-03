from glob import glob
from keras.utils.np_utils import to_categorical
from os import path, listdir
import tifffile as tif
import numpy as np
import pandas as pd
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
