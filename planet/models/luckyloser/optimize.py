# Random hyperparameter optimization for LuckyLoser model.
# Run: while true; do CUDA_VISIBLE_DEVICES="0" python optimize.py; sleep 1; done
from keras.callbacks import Callback
from time import time
import json
import numpy as np
import os
import tensorflow as tf

import sys
sys.path.append('.')
from planet.models.luckyloser.luckyloser import LuckyLoser, outputs, losses, nets


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


class MetricDeadline(Callback):

    def __init__(self, metric_key='F2', nb_epochs=10, min_value=0.7):
        super(Callback, self).__init__()
        self.metric_key = metric_key
        self.nb_epochs = nb_epochs
        self.min_value = min_value
        self.log_values = []

    def on_epoch_end(self, epoch, logs):
        self.log_values.append(logs[self.metric_key])
        self.model.stop_training = (epoch >= self.nb_epochs) and (max(self.log_values) < self.min_value)


def powerball(CUDA_VISIBLE_DEVICES='0', nb_max_hours=4., trn_prop_data=0.1):

    rng = np.random
    seed = int(time()) + os.getpid()
    np.random.seed(seed)
    tf.set_random_seed(seed)

    model = LuckyLoser()
    model.cfg['seed'] = seed

    # JPG or TIFF. TODO: TIFFs
    model.cfg['hdf5_path_trn'] = rng.choice(['data/train-jpg.hdf5'])
    model.cfg['hdf5_path_tst'] = model.cfg['hdf5_path_trn'].replace('train', 'test')

    # Input shape.
    _ = int(rng.choice(np.arange(64, 150, 2)))
    model.cfg['input_shape'] = (_, _, 3 if 'jpg' in model.cfg['hdf5_path_trn'] else 4)

    # Network setup.
    model.cfg['net_builder_func'] = rng.choice(nets)
    model.cfg['net_out_func'] = rng.choice(outputs)
    model.cfg['net_loss_func'] = rng.choice(losses)
    model.cfg['net_threshold'] = rng.choice(np.arange(0.4, 0.5, 0.01))

    # Training setup.
    model.cfg['trn_augment_max_trn'] = rng.randint(0, 20)
    model.cfg['trn_batch_size'] = rng.choice(np.arange(8, 64, 8))
    model.cfg['trn_adam_params'] = {'lr': 10**rng.uniform(-4, -2)}
    model.cfg['trn_prop_trn'] = rng.uniform(0.7, 0.8)
    model.cfg['trn_epochs'] = 1000

    # Optionally decrease amount of data and early stopping.
    model.cfg['trn_prop_data'] = 0.15
    model.cfg['trn_monitor_val'] = False

    # Training returns history object.
    history = model.train(callbacks=[
        MetricDeadline('F2', 10, 0.8),
        MetricDeadline('F2', 60, 0.92),
        TimeoutCB(nb_max_hours)
    ])

    # Rename directory with maximum val_F2 divided by number of epochs.
    nb_epochs = len(history['loss'])
    cpdir = model.cfg['cpdir'].replace(str(seed), '%.3lf_%.4lf_%d' % (
        model.cfg['trn_prop_data'], np.max(history['val_F2']), nb_epochs))
    os.rename(model.cfg['cpdir'], cpdir)

    # Save config to disk.
    payload = serialize_config(model.cfg)
    with open('%s/report.json' % cpdir, 'w') as fp:
        json.dump(payload, fp, indent=4)

    print('Complete - saved results to %s' % cpdir)

if __name__ == "__main__":

    assert 'CUDA_VISIBLE_DEVICES' in os.environ
    assert len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) == 1
    powerball(os.environ['CUDA_VISIBLE_DEVICES'], nb_max_hours=0.9)
