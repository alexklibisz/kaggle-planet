import h5py
import logging
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS
from planet.model_runner import optimize_submit
from time import time


def model_ensemble(ensemble_def_file, hdf5_path_trn='data/train-jpg.hdf5', hdf5_path_tst='data/test-jpg.hdf5'):
    df = pd.read_csv(ensemble_def_file)

    cp_paths = df['checkpoint_path']
    yp_trn_fnames = df['yp_trn_fname']
    yp_tst_fnames = df['yp_test_fname']

    yp_trn = np.zeros((40479, len(TAGS)))
    for cp_path, yp_path in zip(cp_paths, yp_trn_fnames):
        yp_path = yp_path.strip()
        yp = np.load('%s/%s' % (cp_path, yp_path))
        yp_trn += yp/len(yp_trn_fnames)

    yp_tst = np.zeros((61191, len(TAGS)))
    for cp_path, yp_path in zip(cp_paths, yp_tst_fnames):
        yp_path = yp_path.strip()
        yp = np.load('%s/%s' % (cp_path, yp_path))
        yp_tst += yp/len(yp_tst_fnames)

    # Get yt values from trn hdf5 file
    data_trn = h5py.File(hdf5_path_trn)
    yt_trn = data_trn.get('tags')[...]

    # Get image names from hdf5 files
    data_tst = h5py.File(hdf5_path_tst)
    names_trn = data_trn.attrs['names'].split(',')
    names_tst = data_tst.attrs['names'].split(',')

    cpdir = 'checkpoints/ensemble_%d_%d' % (int(time()), os.getpid())
    if not os.path.exists(cpdir):
        os.mkdir(cpdir)
    optimize_submit(cpdir, names_trn, names_tst, yt_trn, yp_trn, yp_tst)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    assert len(sys.argv) == 2, "python model_ensemble.py [path_to_ensemble_csv]" 
    assert os.path.exists(sys.argv[1]), "ensemble csv file not found"

    model_ensemble(sys.argv[1])