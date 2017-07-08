from os import path, listdir
from scipy.stats import entropy
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import h5py
import tifffile as tif
import numpy as np
import pandas as pd
import pickle as pkl
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
IMG_MEAN_JPG_TRN = (79.4163863379, 86.8384090525, 76.201647143)
IMG_STDV_JPG_TRN = ()
IMG_MEAN_TIF_TRN = (4988.75696302, 4270.74552695, 3074.87909779, 6398.84897763)


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


def get_train_val_idxs(tags, prop_data=1.0, prop_trn=0.8, rng=None, nb_iter=2000):
    '''Picks the random training and validation indexes from the given array of tags
    that minimizes the mean absolute error relative the full dataset.'''
    if rng is None:
        rng = np.random

    dist_full = np.sum(tags, axis=0) / len(tags)
    best, min_mae = None, 1e10
    idxs = np.arange(tags.shape[0])
    nbtrn = int(tags.shape[0] * prop_data * prop_trn)
    nbval = int(tags.shape[0] * prop_data * (1 - prop_trn))
    for _ in range(nb_iter):
        rng.shuffle(idxs)
        idxs_trn, idxs_val = idxs[:nbtrn], idxs[-nbval:]
        assert set(idxs_trn).intersection(idxs_val) == set([])
        dist_trn = np.sum(tags[idxs_trn], axis=0) / len(idxs_trn)
        dist_val = np.sum(tags[idxs_val], axis=0) / len(idxs_val)
        if np.count_nonzero(dist_val) < dist_val.shape[0] or np.count_nonzero(dist_trn) < dist_trn.shape[0]:
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


def serialize_config(config):
    _config = {}
    for k, v in config.items():
        if callable(v):
            _config[k] = v.__name__
        elif type(v) == np.int64:
            _config[k] = float(v)
        else:
            _config[k] = v
    return _config


def val_plot_metrics(pickle_path):
    import matplotlib.pyplot as plt

    data = pkl.load(open(pickle_path, "rb"))

    # Plot each tag's F2, precision, and recall over time

    mymin = 1
    tags = sorted(data['tag_metrics'])
    for tag in tags:
        for metric in data['tag_metrics'][tag]:
            _ = min(data['tag_metrics'][tag][metric])
            if _ < mymin:
                mymin = _

    for tag in tags:
        print(tag)
        plt.figure(figsize=(15, 8))
        for metric in data['tag_metrics'][tag]:
            plt.plot(data['tag_metrics'][tag][metric])
            _ = min(data['tag_metrics'][tag][metric])
            if _ < mymin:
                mymin = _
        plt.legend([metric for metric in data['tag_metrics'][tag]], loc='lower right')
        plt.title(tag)
        plt.ylim([mymin, 1])

        plt.show()

    return


def val_plot_predictions(pickle_path):

    import matplotlib.pyplot as plt
    darkgreen = '#196419'
    barwidth = 0.45

    data = pkl.load(open(pickle_path, "rb"))

    yp = np.asarray(data['yp'])
    yt = np.asarray(data['yt'])

    # Sum up for each row, so that we have fn and fp for each tag
    fn = np.sum(np.clip(yt - yp, 0, 1), axis=0)
    fp = np.sum(np.clip(yp - yt, 0, 1), axis=0)

    xvals = np.arange(yp.shape[1])

    plt.figure(figsize=(15, 8))
    plt.bar(xvals, fp, label="False Positives", color=darkgreen)
    plt.xticks(xvals, TAGS_short)
    plt.title("False Positives")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.bar(xvals, fn, label="False Negatives", color='black')
    plt.xticks(xvals, TAGS_short)
    plt.title("False Negatives")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.bar(xvals - barwidth / 2, fp, barwidth, label="False Positives", color=darkgreen)
    plt.bar(xvals + barwidth / 2, fn, barwidth, label="False Negatives", color='black')
    plt.xticks(xvals, TAGS_short)
    plt.title("False Positives Compared To False Negatives")
    plt.legend()
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.bar(xvals, fp + fn, color=(0.6, 0.1, 0.1))
    plt.xticks(xvals, TAGS_short)
    plt.title("Total Incorrect")
    plt.legend()
    plt.ylabel("Count")
    plt.show()

    numCorrect = np.sum(np.equal(yt, yp), axis=1)

    print("I'm sorry if the x-axis labels are all shifted messed. I'll try to fix that eventually")

    plt.figure(figsize=(15, 8))
    plt.hist(len(TAGS) - numCorrect, len(TAGS), rwidth=0.9, log=True)
    plt.title("Number of Incorrect Labels Histogram (Log Scale)")
    plt.xticks(list(range(len(TAGS))))
    plt.ylabel("Count")
    plt.xlabel("Number of Incorrect Labels")

    plt.figure(figsize=(15, 8))
    plt.hist(len(TAGS) - numCorrect, len(TAGS), rwidth=0.9)
    plt.title("Number of Incorrect Labels Histogram (Linear Scale)")
    plt.xticks(list(range(len(TAGS))))
    plt.ylabel("Count")
    plt.xlabel("Number of Incorrect Labels")

    k = 10
    if k > len(numCorrect) - 1:
        k = len(numCorrect) - 1

    fig, axes = plt.subplots(k, figsize=(15, 8 * k))
    xvals = np.arange(yp.shape[1])

    # Get and sort the k lowest indices
    kSmallest = np.argpartition(numCorrect, k)[:k]
    kSmallest = sorted(kSmallest, key=lambda i1: numCorrect[i1])

    colors = [0] * len(TAGS)
    for idx, ax in zip(kSmallest, fig.axes):
        points = yp[idx] - yt[idx]
        for i in range(len(points)):
            if yp[idx][i] > yt[idx][i]:
                colors[i] = '#196419'
            elif yp[idx][i] < yt[idx][i]:
                colors[i] = 'black'
            else:
                colors[i] = 'b'

        ax.scatter(xvals, points, marker='x', c=colors)
        ax.set_xticks(xvals)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["false negative", "correct", "false positive"])
        ax.set_xticklabels(TAGS_short)
    plt.show()
    return


def f2pr(yt, yp, thresholds):
    yp = (yp > thresholds).astype(np.uint8)

    tp = np.sum(yt * yp, axis=0)
    fp = np.sum(np.clip(yp - yt, 0, 1), axis=0)
    fn = np.sum(np.clip(yt - yp, 0, 1), axis=0)

    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)
    b = 2.0
    f2 = (1 + b**2) * ((p * r) / (b**2 * p + r + 1e-7))
    return f2, p, r


# Given a matrix of yt and yp, where each row is a separate prediction and each column
# is for a separate tag, this returns F2 scores, recall, and precision for each tag
# using the given thresholds
def tags_f2pr(yt, yp, thresholds):
    yp = (yp > thresholds).astype(np.uint8)
    return f2pr(yt, yp, axis=0)

# Given a matrix of yt and yp, where each row is a separate prediction and each column
# is for a separate tag, this returns a F2 scores, recall, and precision for each tag
# for each threshold in the thresholds_to_try array


def _tags_f2pr(yt, yp, thresholds_to_try):
    yp = (yp > thresholds_to_try[:, None, None]).astype(np.uint8)
    yt = yt[None, :, :]
    return f2pr(yt, yp, axis=1)

# Given a matrix of yt and yp, where each row is a separate prediction and each column
# is a separate label, this returns  F2 scores, one for each label
# Note: this assumes you've already taken care of the thresholding for yp


def f2pr(yt, yp, axis=None):
    tp = np.sum(yt * yp, axis=axis)
    fp = np.sum(np.clip(yp - yt, 0, 1), axis=axis)
    fn = np.sum(np.clip(yt - yp, 0, 1), axis=axis)

    # p = tp/(tp + fp + 1e-1)
    # r = tp/(tp + fn + 1e-1)

    p = np.divide(tp, (np.add(np.add(tp, fp, dtype=np.int32), 1e-7, dtype=np.float32)), dtype=np.float32)
    r = np.divide(tp, (np.add(np.add(tp, fn, dtype=np.int32), 1e-7, dtype=np.float32)), dtype=np.float32)
    b = 2.0
    f2 = (1 + b**2) * ((p * r) / (b**2 * p + r + 1e-7))
    return f2, p, r


def optimize_thresholds(yt, yp, n=101):
    thresholds_to_try = np.linspace(0, 1, n)
    f2, _, _ = _tags_f2pr(yt, yp, thresholds_to_try)

    # Find the thresholds that gave the highest f2 score
    best_indices = np.argmax(f2, axis=0)
    return thresholds_to_try[best_indices]
