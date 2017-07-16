from glob import glob
import h5py
import logging
import os
import numpy as np
import pandas as pd
import pickle
import sys
from itertools import permutations
from hyperopt import fmin, hp, space_eval, tpe, STATUS_OK, Trials
from math import ceil
from multiprocessing import Pool
from operator import itemgetter
from time import time
from tqdm import tqdm
sys.path.append('.')
from planet.utils.data_utils import TAGS, TAGS_short, optimize_thresholds, f2pr
from planet.utils.runtime import funcname
from planet.model_runner import submission
np.random.seed(int(time()))
NUM_IMAGES_TRN = 40479
NUM_IMAGES_TST = 61191
NUM_OUTPUTS = len(TAGS)

TIME = int(time())
ENSEMBLE_DIR = 'ensembles'

def optimize_single_tag(args):
    """ Takes two slices of arrays for a single tag and returns the optimized weights"""
    (tag_idx, yp_trn, yt_trn, nb_iter) = args
    N_trn = yp_trn.shape[0]

    tag_trials = Trials()
    def objective(w, yt_trn=yt_trn, yp_trn=yp_trn):
        f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, yp_trn, w)
        if len(tag_trials.trials) > 1:
            if f2_opt > -tag_trials.best_trial['result']['loss']:
                logger = logging.getLogger(funcname())
                logger.info('%2d - %s: hyperopt f2 improved from %lf to %lf' 
                    % (tag_idx, TAGS_short[tag_idx], -tag_trials.best_trial['result']['loss'], f2_opt))
        if len(tag_trials.trials) % ceil(nb_iter/10) == 0:
                logger = logging.getLogger(funcname())
                logger.info('%2d - %s: %d/%d: hyperopt f2 is %lf'
                 % (tag_idx, TAGS_short[tag_idx], len(tag_trials.trials), nb_iter, -tag_trials.best_trial['result']['loss']))
        return {
            'loss':-f2_opt,
            'status': STATUS_OK,
            'thresholds': thresh_opt,
            'weights': w,
        }
    # Build search space
    weights_space_tag = [hp.uniform(str(i), 0, 1) for i in range(N_trn)]
    total = 0
    for s in weights_space_tag:
        total += s
    for i in range(len(weights_space_tag)):
        weights_space_tag[i] /= total

    best = fmin(objective, space=weights_space_tag, algo=tpe.suggest, max_evals=nb_iter, trials=tag_trials)
    return space_eval(weights_space_tag, best)


def get_weighted_yp(w, yp_all):
    # TODO: remove for-loop and only use matrix operations
    yp = np.zeros(yp_all.shape[1:])
    for w_mem, yp_mem in zip(w, yp_all):
        yp += w_mem*yp_mem
    return yp


def get_weighted_optimized_results(yt_trn, yp_trn, w):
    w /= np.sum(w, axis=0)             # Normalize weights
    yp_trn = get_weighted_yp(w,yp_trn) # Get weighted average for yp
    thresh_opt = optimize_thresholds(yt_trn, yp_trn)
    f2_opt, p_opt, r_opt = f2pr(yt_trn, (yp_trn > thresh_opt).astype(np.uint8))
    return f2_opt, thresh_opt, w

class EnsembleOptimizer(object):
    def __init__(self, ensemble_def_file):
        logger = logging.getLogger(funcname())

        # Read spreadsheet values to get prediction paths.
        df = pd.read_csv(ensemble_def_file) 
        paths_yp_trn_initial = []
        for n, gtrn in enumerate(df['yp_trn_glob'].values):
            paths = sorted(glob(os.path.expanduser(gtrn)))
            if len(paths) == 0:
                logger.warn("No files found for glob %d: %s" % (n, gtrn))
            paths_yp_trn_initial += paths 

        self.paths_yp_trn = []

        # Read and concatenate all of the yp_trn and yp_tst files into two matrices.
        logger.info("Loading predictions...")
        self.yp_trn_all = []
        self.yp_tst_all = []
        for i, p_trn in enumerate(paths_yp_trn_initial):
            logger.info(p_trn)

            p_tst = p_trn.replace('trn', 'tst')
            if not os.path.exists(p_tst):
                logger.warning("Skipping %s because it doesn't have a matching test file." % p_trn)
                continue

            yp_trn = np.load(p_trn)
            if yp_trn.shape != (NUM_IMAGES_TRN, NUM_OUTPUTS):
                logger.warning("Skipping %s because the trn shape is incorrect" % p_trn)
                continue

            yp_tst = np.load(p_tst)
            if yp_tst.shape != (NUM_IMAGES_TST, NUM_OUTPUTS):
                logger.warning("Skipping %s because the tst shape is incorrect" % p_tst)
                continue

            self.yp_trn_all.append(yp_trn)
            self.yp_tst_all.append(yp_tst)
            self.paths_yp_trn.append(p_trn)
        self.yp_trn_all = np.array(self.yp_trn_all)
        self.yp_tst_all = np.array(self.yp_tst_all)

    def fit(self, yt_trn, nb_iter=5, use_hyperopt=True, use_rand_search=False, max_parallel=4):
        logger = logging.getLogger(funcname())
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        N_trn = len(self.paths_yp_trn)

        # results format: [(F2 score, tag thresholds, ensemble weights)]
        results, best_idx = [], 0

        # Try equal weights
        w = np.ones((N_trn, NUM_OUTPUTS), dtype=np.float16)
        f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, self.yp_trn_all, w)
        results.append((f2_opt, thresh_opt, w))
        logger.info('Equal weights: f2 = %f' % f2_opt)


        # Try using each member individually.
        try_individually = False
        if try_individually:
            w = np.zeros((N_trn, NUM_OUTPUTS), dtype=np.float16)
            for it in range(N_trn):
                w *= 0
                w[it] = 1
                f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, self.yp_trn_all, w)
                logger.info('%-70s f2 = %f' % (self.paths_yp_trn[it][-60:], f2_opt))
                if f2_opt > 0.88:
                    results.append((f2_opt, thresh_opt, w))
                    if f2_opt > results[best_idx][0]:
                        old_best = results[best_idx][0]
                        best_idx = len(results)-1
                        logger.info('f2 improved from %lf to %lf' % (old_best, f2_opt))
                        # serialize_ensemble(f2_opt, thresh_opt, w, names_tst, yp_tst_all, self.yp_trn_all, self.paths_yp_trn)

        if use_hyperopt:
            optimize_individually = True
            if optimize_individually:
                logging.getLogger("hyperopt.tpe").setLevel(logging.WARNING)
                if max_parallel == 1:
                    tag_weights = []
                    for tag_idx in range(NUM_OUTPUTS):
                        logger.info("Optimizing tag %d" % tag_idx)
                        tag_weights.append(optimize_single_tag(tag_idx, self.yp_trn_all[:,:,tag_idx], yt_trn[:, tag_idx], nb_iter))
                else:
                    p = Pool(min(max_parallel, NUM_OUTPUTS))
                    tag_weights = p.map(optimize_single_tag, [(tag_idx, self.yp_trn_all[:,:,tag_idx], yt_trn[:, tag_idx], nb_iter) for tag_idx in range(NUM_OUTPUTS)])
                w = np.transpose(tag_weights)
                f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, self.yp_trn_all, w)
                logger.info("Optimized weights: f2 = %f" % f2_opt)
                results.append((f2_opt, thresh_opt, w))
            else:
                self.trials = Trials()
                def objective(w, self=self, yt_trn=yt_trn):
                    w = np.array(w).reshape((N_trn, NUM_OUTPUTS))
                    f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, self.yp_trn_all, w)
                    if len(self.trials.trials) > 1:
                        if f2_opt > -self.trials.best_trial['result']['loss']:
                            logger = logging.getLogger(funcname())
                            logger.info('hyperopt f2 improved from %lf to %lf' % (-self.trials.best_trial['result']['loss'], f2_opt))
                    return {
                        'loss':-f2_opt,
                        'status': STATUS_OK,
                        'thresholds': thresh_opt,
                        'weights': w,
                    }
                weights_space_all = [hp.uniform(str(i), 0, 1) for i in range(N_trn*NUM_OUTPUTS)]
                total = 0
                for s in weights_space_all:
                    total += s
                for i in range(len(weights_space_all)):
                    weights_space_all[i] /= total

                best = fmin(objective, space=weights_space_all, algo=tpe.suggest, max_evals=nb_iter, trials=self.trials)
                sortedTrials = sorted(self.trials.trials, key=lambda x: x['result']['loss'])[:10]

                # Process hyperopt results into our results list
                for t in sortedTrials:
                    f2_opt = -t['result']['loss']
                    thresh_opt = t['result']['thresholds']
                    w = t['result']['weights']
                    if f2_opt > 0.88:
                        results.append((f2_opt, thresh_opt, w))
                        if f2_opt > results[best_idx][0]:
                            best_idx = len(results)-1
        if use_rand_search:
            # Randomly choose a weight for each tag across all member predictions.
            logger.info('Searching random weights...')
            f2_scores = []
            for it in range(nb_iter):
                w = np.random.rand(N_trn, NUM_OUTPUTS)
                f2_opt, thresh_opt, w = get_weighted_optimized_results(yt_trn, self.yp_trn_all, w)
                f2_scores.append(f2_opt)
                if f2_opt > 0.88:
                    results.append((f2_opt, thresh_opt, w))
                    if f2_opt > results[best_idx][0]:
                        old_best = results[best_idx][0]
                        best_idx = len(results)-1
                        logger.info('%-05.2lf: f2 improved from %lf to %lf:\n' % ((it / nb_iter * 100), old_best, f2_opt))
                        # serialize_ensemble(f2_opt, thresh_opt, w, names_tst, yp_tst_all, self.yp_trn_all, self.paths_yp_trn)

                if it % 50 == 0:
                    logger.info('%-05.2lf: f2 mean=%.4lf, min=%.4lf, max=%.4lf, stdv=%.4lf, unique=%d' % \
                         ((it / nb_iter * 100), np.mean(f2_scores), np.min(f2_scores), np.max(f2_scores), 
                            np.std(f2_scores), len(np.unique(f2_scores))))

        return results

if __name__ == "__main__":

    # import resource, sys
    # resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    # sys.setrecursionlimit(10**7)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    assert len(sys.argv) == 3, "python model_ensemble.py path_to_ensemble_csv nb_iter"
    assert os.path.exists(sys.argv[1]), "ensemble csv file not found"

    if not os.path.exists(ENSEMBLE_DIR):
        os.mkdir(ENSEMBLE_DIR)

    ens_opt = EnsembleOptimizer(sys.argv[1])

    hdf5_path_trn='data/train-jpg.hdf5'
    hdf5_path_tst='data/test-jpg.hdf5'

    # Read ground-truth values.
    data_trn = h5py.File(hdf5_path_trn, 'r')
    yt_trn = data_trn.get('tags')[...]

    # Get image names from hdf5 file
    data_tst = h5py.File(hdf5_path_tst, 'r')
    names_tst = data_tst.attrs['names'].split(',')

    results = ens_opt.fit(yt_trn, nb_iter=int(sys.argv[2]))#, max_parallel=NUM_OUTPUTS)

    # Make submissions with best 10 results.
    results = sorted(results, key=lambda x: x[0], reverse=True)[:10]
    for f2, thresh_opt, w in results:
        fstem = '%s/ensemble_%.6f_%d' % (ENSEMBLE_DIR, f2, TIME)
        logger.info('Making submission with f2 = %lf' % f2)
        yp_tst = get_weighted_yp(w, ens_opt.yp_tst_all)
        yp_tst_opt = (yp_tst > thresh_opt).astype(np.uint8)
        submission(names_tst, yp_tst_opt, '%s.csv' % fstem)

        f = h5py.File('%s.hdf5' % fstem, 'w')
        f.attrs['paths_to_npy_files'] = ','.join(ens_opt.paths_yp_trn)

        ds = f.create_dataset('weights', w.shape, dtype=w.dtype)
        ds[...] = w

        ds = f.create_dataset('thresholds', thresh_opt.shape, dtype=thresh_opt.dtype)
        ds[...] = thresh_opt

        ds = f.create_dataset('yp_tst_cmb', yp_tst.shape, dtype=yp_tst.dtype)
        ds[...] = yp_tst

        yp_trn = get_weighted_yp(w, ens_opt.yp_trn_all)
        ds = f.create_dataset('yp_trn_cmb', yp_trn.shape, dtype=yp_trn.dtype)
        ds[...] = yp_trn
        f.close()

