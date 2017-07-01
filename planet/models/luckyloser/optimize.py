# Random hyperparameter optimization for LuckyLoser model.
# Run: while true; do CUDA_VISIBLE_DEVICES="0" python optimize.py; sleep 1; done
from keras.callbacks import Callback
from time import time
import json
import numpy as np
import os

import sys
sys.path.append('.')
from planet.models.luckyloser.luckyloser import LuckyLoser, _out_single_sigmoid, _loss_bcr, _net_vgg16, _net_vgg16_pretrained, _net_vgg19, _net_vgg19_pretrained


def serialize_config(config):
    for k, v in config.items():
        if callable(v):
            config[k] = v.__name__
        if type(v) == np.int64:
            config[k] = float(v)
    return config


class TimeoutCB(Callback):

    def __init__(self, nb_max_hours):
        super(Callback, self).__init__()
        self.time_max = time() + (nb_max_hours * 60 * 60)

    def on_epoch_end(self, epoch, logs):
        self.model.stop_training = time() > self.time_max


def powerball(CUDA_VISIBLE_DEVICES='0', nb_max_hours=4., trn_prop_data=0.1):

    ID = int(time()) + os.getpid()
    rng = np.random
    rng.seed(ID)

    model = LuckyLoser()
    model.cfg['cpdir'] = 'checkpoints/luckyloser_rs_%d' % ID

    # JPG or TIFF. TODO: TIFFs
    model.cfg['hdf5_path_trn'] = rng.choice(['data/train-jpg.hdf5'])
    model.cfg['hdf5_path_tst'] = model.cfg['hdf5_path_trn'].replace('train', 'test')

    # Input shape.
    hw = rng.randint(64, 256)
    model.cfg['input_shape'] = (hw, hw, 3 if 'jpg' in model.cfg['hdf5_path_trn'] else 4)

    # Network setup.
    model.cfg['net_builder_func'] = rng.choice([_net_vgg16, _net_vgg16_pretrained, _net_vgg19, _net_vgg19_pretrained])
    model.cfg['net_out_func'] = rng.choice([_out_single_sigmoid])
    model.cfg['net_loss_func'] = rng.choice([_loss_bcr])

    # Training setup.
    model.cfg['trn_augment_max_trn'] = rng.randint(0, 20)
    model.cfg['trn_batch_size'] = rng.choice(np.arange(10, 100, 2))
    model.cfg['trn_adam_params'] = {'lr': 10**rng.uniform(-4, -2)}
    model.cfg['trn_prop_trn'] = rng.uniform(0.7, 0.8)
    model.cfg['trn_epochs'] = 1000

    # Optionally decrease amount of data and early stopping.
    model.cfg['trn_prop_data'] = 0.07
    model.cfg['trn_early_stop'] = False

    # Training returns history object.
    history = model.train(callbacks=[TimeoutCB(nb_max_hours)])

    # Rename directory with maximum val_F2 divided by number of epochs.
    nb_epochs = len(history['loss'])
    cpdir = model.cfg['cpdir'].replace(str(ID), '%.4lf_%d' % (np.max(history['val_F2']), nb_epochs))
    os.rename(model.cfg['cpdir'], cpdir)

    # Save config to disk.
    payload = serialize_config(model.cfg)
    with open('%s/report.json' % cpdir, 'w') as fp:
        json.dump(payload, fp, indent=4)

if __name__ == "__main__":

    assert 'CUDA_VISIBLE_DEVICES' in os.environ
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    powerball(os.environ['CUDA_VISIBLE_DEVICES'], nb_max_hours=4.)
