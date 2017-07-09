from glob import glob
import h5py
import logging
import os
import numpy as np
import pandas as pd
import sys
from itertools import permutations
from operator import itemgetter
from time import time
from tqdm import tqdm
sys.path.append('.')
from planet.utils.data_utils import TAGS, optimize_thresholds, f2pr
from planet.utils.runtime import funcname
from planet.model_runner import submission

NUM_IMAGES_TRN = 40479
NUM_IMAGES_TST = 61191
NUM_OUTPUTS = len(TAGS)

def model_ensemble(ensemble_def_file, hdf5_path_trn='data/train-jpg.hdf5', hdf5_path_tst='data/test-jpg.hdf5', nb_iter=2000):
    logger = logging.getLogger(funcname())

    # Read spreadsheet values to get prediction paths.
    df = pd.read_csv(ensemble_def_file)
    cpdirs = df['checkpoint_dir'].values
    globs_yp_trn = ['%s/%s' % (os.path.expanduser(d).strip(), g.strip())
                    for d, g in zip(cpdirs, df['yp_trn_glob'].values)]
    globs_yp_tst = ['%s/%s' % (os.path.expanduser(d).strip(), g.strip())
                    for d, g in zip(cpdirs, df['yp_tst_glob'].values)]
    paths_yp_trn, paths_yp_tst = [], []
    for gtrn, gtst in zip(globs_yp_trn, globs_yp_tst):
        paths_yp_trn += sorted(glob(gtrn))
        paths_yp_tst += sorted(glob(gtst))

    # Read and concatenate all of the yp_trn and yp_tst files into two matrices.
    N_trn, N_tst = len(paths_yp_trn), len(paths_yp_tst)
    yp_trn_all = np.zeros((N_trn, NUM_IMAGES_TRN, NUM_OUTPUTS), dtype=np.float16)
    yp_tst_all = np.zeros((N_tst, NUM_IMAGES_TST, NUM_OUTPUTS), dtype=np.float16)
    for i, p in enumerate(paths_yp_trn):
        yp_trn_all[i] = np.load(p)
    for i, p in enumerate(paths_yp_tst):
        yp_tst_all[i] = np.load(p)

    # Read ground-truth values.
    data_trn = h5py.File(hdf5_path_trn, 'r')
    yt_trn = data_trn.get('tags')[...]

    def weighted_yp(w, yp_all):
        # TODO: remove for-loop and only use matrix operations
        yp = np.zeros((NUM_IMAGES_TRN, NUM_OUTPUTS))
        for w_idv, yp_idv in zip(w, yp_trn_all):
            yp += w_idv*yp_idv
        return yp
        # return w[:,None,:]*yp_trn_all

    def get_answers(yt_trn, yp_trn_all, w):
        # Normalize weights
        w /= np.sum(w, axis=0)

        # Get weighted average for yp
        yp_trn = weighted_yp(w, yp_trn_all)

        thresh_opt = optimize_thresholds(yt_trn, yp_trn)
        f2_opt, p_opt, r_opt = f2pr(yt_trn, (yp_trn > thresh_opt).astype(np.uint8))
        return f2_opt, thresh_opt, w

    top_stuff = []
    best_idx = 0

    # Try equal weights
    w = np.ones((N_trn, NUM_OUTPUTS), dtype=np.float16)
    f2_opt, thresh_opt, w = get_answers(yt_trn, yp_trn_all, w)
    top_stuff.append((f2_opt, thresh_opt, w))
    logger.info('Equal weights: f2 = %f: %s' % (f2_opt, w))

    # Try using each member individually
    w = np.zeros((N_trn, NUM_OUTPUTS), dtype=np.float16)
    ones_idxs = [0]*NUM_OUTPUTS
    for it in tqdm(range(N_trn)):
        w[it] = 1
        f2_opt, thresh_opt, w = get_answers(yt_trn, yp_trn_all, w)
        print('%s: f2 = %f' % (paths_yp_trn[it], f2_opt))
        w[it] = 0
        if f2_opt > 0.88:
            top_stuff.append((f2_opt, thresh_opt, w))
            if f2_opt > top_stuff[best_idx][0]:
                old_best = top_stuff[best_idx][0]
                best_idx = len(top_stuff)-1
                logger.info('   f2 improved from %lf to %lf' % (it, old_best, f2_opt))

    # Randomly choose a weight for each tag across all member predictions.
    # The weights for each tag sum to 1.
    for it in tqdm(range(nb_iter)):
        w = np.random.rand(N_trn, NUM_OUTPUTS)
        f2_opt, thresh_opt, w = get_answers(yt_trn, yp_trn_all, w)
        if f2_opt > 0.88:
            top_stuff.append((f2_opt, thresh_opt, w))
            if f2_opt > top_stuff[best_idx][0]:
                old_best = top_stuff[best_idx][0]
                best_idx = len(top_stuff)-1
                logger.info('%3d: f2 improved from %lf to %lf: %s' % (it, old_best, f2_opt, w))


    # Get image names from hdf5 file
    data_tst = h5py.File(hdf5_path_tst, 'r')
    names_tst = data_tst.attrs['names'].split(',')

    # Make submissions with best 10 stuff
    top_stuff.sort(key=itemgetter(0))
    top_stuff = top_stuff[:10]
    t = time()

    # Checkpoint dir for this ensemble.
    cpdir = 'checkpoints/ensemble_%d' % (int(time()))
    if not os.path.exists(cpdir):
        os.mkdir(cpdir)

    for idx, (f2, thresh_opt, w) in enumerate(top_stuff):
        logger.info('Making submission with f2 = %lf: %s' % (f2, w))

        # Get weighted average for yp
        yp_tst = weighted_yp(w, yp_tst_all)

        yp_tst_opt = (yp_tst > thresh_opt).astype(np.uint8)
        stem = '%s/submission_tst_opt_%d_%.6f_%02d' % (cpdir, t, f2, idx)

        np.savez('%s.npz'%stem, weights=w, thresholds=thresh_opt)
        submission(names_tst, yp_tst_opt, '%s.csv' % stem)

        # Save raw activations.
        # np_path = '%s/yp_tst_%d.npy' % (cpdir, t)
        # np.save(np_path, yp_tst)
        # logger.info('Saved %s.' % np_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    assert len(sys.argv) == 2, "python model_ensemble.py [path_to_ensemble_csv]"
    assert os.path.exists(sys.argv[1]), "ensemble csv file not found"

    model_ensemble(sys.argv[1])
