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

np.random.seed(int(time()))
NUM_IMAGES_TRN = 40479
NUM_IMAGES_TST = 61191
NUM_OUTPUTS = len(TAGS)

def model_ensemble(ensemble_def_file, hdf5_path_trn='data/train-jpg.hdf5', hdf5_path_tst='data/test-jpg.hdf5', nb_iter=5):
    logger = logging.getLogger(funcname())
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    # Read spreadsheet values to get prediction paths.
    df = pd.read_csv(ensemble_def_file) 
    paths_yp_trn, paths_yp_trn_valid = [], []
    for gtrn in df['yp_trn_glob'].values:
        paths_yp_trn += sorted(glob(os.path.expanduser(gtrn)))

    # Read and concatenate all of the yp_trn and yp_tst files into two matrices.
    logger.info("Loading predictions...")
    yp_trn_all, yp_tst_all = [], []
    for i, p_trn in enumerate(paths_yp_trn):
        logger.info(p_trn)
        p_tst = p_trn.replace('trn', 'tst')
        if not os.path.exists(p_tst):
            logger.warning("Skipping %s because it doesn't have a matching test file." % p_trn)
            continue
        yp_trn_all.append(np.load(p_trn))
        yp_tst_all.append(np.load(p_tst))
        paths_yp_trn_valid.append(p_trn)

    N_trn = len(paths_yp_trn_valid)
    yp_trn_all = np.array(yp_trn_all)
    yp_tst_all = np.array(yp_tst_all)

    # Read ground-truth values.
    data_trn = h5py.File(hdf5_path_trn, 'r')
    yt_trn = data_trn.get('tags')[...]

    def get_weighted_yp(w, yp_all):
        # TODO: remove for-loop and only use matrix operations
        yp = np.zeros(yp_all.shape[1:])
        for w_mem, yp_mem in zip(w, yp_all):
            yp += w_mem*yp_mem
        return yp

    def get_weighted_optimized_results(yt_trn, yp_trn_all, w):
        w /= np.sum(w, axis=0)                  # Normalize weights
        yp_trn = get_weighted_yp(w, yp_trn_all) # Get weighted average for yp
        thresh_opt = optimize_thresholds(yt_trn, yp_trn)
        f2_opt, p_opt, r_opt = f2pr(yt_trn, (yp_trn > thresh_opt).astype(np.uint8))
        return f2_opt, thresh_opt, w

    # results format: [(F2 score, tag thresholds, ensemble weights)]
    results, best_idx = [], 0

    # Try equal weights
    w = np.ones((N_trn, NUM_OUTPUTS), dtype=np.float16)
    f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, yp_trn_all, w)
    results.append((f2_opt, thresh_opt, w))
    logger.info('Equal weights: f2 = %f' % f2_opt)

    # Try using each member individually.
    w = np.zeros((N_trn, NUM_OUTPUTS), dtype=np.float16)
    for it in range(N_trn):
        w *= 0
        w[it] = 1
        f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, yp_trn_all, w)
        logger.info('%-70s f2 = %f' % (paths_yp_trn_valid[it][-60:], f2_opt))
        if f2_opt > 0.88:
            results.append((f2_opt, thresh_opt, w))
            if f2_opt > results[best_idx][0]:
                old_best = results[best_idx][0]
                best_idx = len(results)-1
                logger.info('f2 improved from %lf to %lf' % (old_best, f2_opt))

    # Randomly choose a weight for each tag across all member predictions.
    logger.info('Searching random weights...')
    f2_scores = []
    for it in range(nb_iter):
        w = np.random.rand(N_trn, NUM_OUTPUTS)
        f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, yp_trn_all, w)
        f2_scores.append(f2_opt)
        if f2_opt > 0.88:
            results.append((f2_opt, thresh_opt, w))
            if f2_opt > results[best_idx][0]:
                old_best = results[best_idx][0]
                best_idx = len(results)-1
                logger.info('%3d: f2 improved from %lf to %lf:\n%s' % (it, old_best, f2_opt, w))
        
        if it % 50 == 0:
            logger.info('%-05.2lf: f2 mean=%.4lf, min=%.4lf, max=%.4lf, stdv=%.4lf, unique=%d' % \
                 ((it / nb_iter * 100), np.mean(f2_scores), np.min(f2_scores), np.max(f2_scores), 
                    np.std(f2_scores), len(np.unique(f2_scores))))

    # Get image names from hdf5 file
    data_tst = h5py.File(hdf5_path_tst, 'r')
    names_tst = data_tst.attrs['names'].split(',')

    # Make submissions with best 10 results.
    results = sorted(results, key=lambda x: x[0], reverse=True)[:10]
    t = time()

    # Checkpoint dir for this ensemble.
    cpdir = 'checkpoints/ensemble_%d' % (int(time()))
    if not os.path.exists(cpdir):
        os.mkdir(cpdir)

    for idx, (f2, thresh_opt, w) in enumerate(results):
        logger.info('Making submission with f2 = %lf' % f2)
        yp_tst = get_weighted_yp(w, yp_tst_all)
        yp_tst_opt = (yp_tst > thresh_opt).astype(np.uint8)
        stem = '%s/submission_tst_opt_%d_%.6f_%02d' % (cpdir, t, f2, idx)
        np.savez('%s.npz'%stem, weights=w, thresholds=thresh_opt)
        submission(names_tst, yp_tst_opt, '%s.csv' % stem)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    assert len(sys.argv) == 3, "python model_ensemble.py path_to_ensemble_csv nb_iter"
    assert os.path.exists(sys.argv[1]), "ensemble csv file not found"

    model_ensemble(sys.argv[1], nb_iter=int(sys.argv[2]))
