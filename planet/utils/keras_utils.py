from keras.callbacks import Callback, TensorBoard
from keras.engine import Layer
from math import ceil
from shutil import copyfile
from sklearn.metrics import fbeta_score, precision_score, recall_score
from termcolor import colored
from time import sleep, ctime
from tqdm import tqdm
import keras.backend as K
import logging
import numpy as np
import os
import pickle as pkl

from planet.utils.data_utils import TAGS, optimize_thresholds, f2pr
from planet.utils.runtime import funcname


def prec(yt, yp):
    yp = K.round(yp)
    tp = K.sum(yt * yp)
    fp = K.sum(K.clip(yp - yt, 0, 1))
    return tp / (tp + fp + K.epsilon())


def reca(yt, yp):
    yp = K.round(yp)
    tp = K.sum(yt * yp)
    fn = K.sum(K.clip(yt - yp, 0, 1))
    return tp / (tp + fn + K.epsilon())


def F2(yt, yp):
    p = prec(yt, yp)
    r = reca(yt, yp)
    b = 2.0
    return (1 + b**2) * ((p * r) / (b**2 * p + r + K.epsilon()))


def tag_metrics():

    # Decorator for procedurally generating functions with different names.
    def rename(newname):
        def decorator(f):
            f.__name__ = newname
            return f
        return decorator

    # Generate an F2 metric for each tag.
    metrics = []
    for i, t in enumerate(TAGS):

        # @rename('%s_acc' % t[:4])
        # def acc(yt, yp, i=i):
        #     return K.sum(K.cast(yt[:, i] == yp[:, i], 'float')) / K.sum(K.clip(yt[:, i], 1, 1))
        # metrics.append(tmp)

        @rename('%s_dif' % t[:4])
        def tmp(yt, yp, i=i):
            t = K.sum(yt[:, i]) / K.sum(K.clip(yt[:, i], 1, 1))
            p = K.sum(yp[:, i]) / K.sum(K.clip(yp[:, i], 1, 1))
            return p - t
        metrics.append(tmp)

        @rename('%s_f2' % t[:4])
        def tmp(yt, yp, i=i):
            return F2(yt[:, i], yp[:, i])
        metrics.append(tmp)

    return metrics


class ThresholdedSigmoid(Layer):

    def __init__(self, lower=0.2, upper=0.8, **kwargs):
        super(ThresholdedSigmoid, self).__init__(**kwargs)
        self.supports_masking = True
        self.lower = K.cast_to_floatx(lower)
        self.upper = K.cast_to_floatx(upper)

    def call(self, inputs):
        return K.clip(K.sigmoid(inputs), self.lower, self.upper)

    def get_config(self):
        config = {'lower': float(self.lower), 'upper': float(self.upper)}
        base_config = super(ThresholdedSigmoid, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class TensorBoardWrapper(TensorBoard):
    '''Sets the self.validation_data property for use with TensorBoard callback.'''

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen
        self.nb_steps = nb_steps

    def on_epoch_end(self, epoch, logs):
        imgs, tags = None, None
        for s in range(self.nb_steps):
            ib, tb = next(self.batch_gen)
            if imgs is None and tags is None:
                imgs = np.zeros((self.nb_steps * ib.shape[0], *ib.shape[1:]), dtype=np.float32)
                tags = np.zeros((self.nb_steps * tb.shape[0], *tb.shape[1:]), dtype=np.uint8)
            imgs[s * ib.shape[0]:(s + 1) * ib.shape[0]] = ib
            tags[s * tb.shape[0]:(s + 1) * tb.shape[0]] = tb
        self.validation_data = [imgs, tags, np.ones(imgs.shape[0]), 0.0]
        return super().on_epoch_end(epoch, logs)


class ValidationCB(Callback):
    '''Compute validation metrics and save data to disk for visualizing/exploring.'''

    def __init__(self, cpdir, batch_gen, batch_size, nb_steps):
        super(Callback, self).__init__()
        self.cpdir = cpdir
        self.batch_size = batch_size
        self.batch_gen = batch_gen
        self.nb_steps = nb_steps
        self.best_metric = 0.
        self.best_epoch = 0.
        assert os.path.exists(self.cpdir)

    def on_epoch_end(self, epoch, logs):
        '''
        1. Predictions on every image in the validation set.
        2. Evaluate the precision, recall, and F2 of each tag.
        3. Store the metrics and predictions in a pickle file in the cpdir.
        '''

        # Make predictions and store all true and predicted tags.
        yt = np.zeros((self.batch_size * self.nb_steps, len(TAGS)))
        yp = np.zeros((self.batch_size * self.nb_steps, len(TAGS)))
        for bidx in tqdm(range(self.nb_steps)):
            ib, tb = next(self.batch_gen)
            yt[bidx * self.batch_size:(bidx + 1) * self.batch_size] = tb
            yp[bidx * self.batch_size:(bidx + 1) * self.batch_size] = self.model.predict_on_batch(ib)

        # Find optimal thresholds.
        thresholds = optimize_thresholds(yt, yp)
        yp_opt = (yp > thresholds).astype(np.uint8)

        # Print per-tag metrics with stress-inducing colors.
        logger = logging.getLogger(funcname())
        for tidx in range(len(TAGS)):
            f2, p, r = f2pr(yt[:, tidx], yp_opt[:, tidx])
            s = '%-20s F2=%.3lf p=%.3lf r=%.3lf t=%.3lf' % (TAGS[tidx], f2, p, r, thresholds[tidx])
            if f2 > 0.9:
                s = colored(s, 'green')
            elif f2 > 0.8:
                s = colored(s, 'yellow')
            else:
                s = colored(s, 'red')
            logger.info(s)

        # Metrics on all predictions.
        f2, p, r = f2pr(yt, (yp > 0.5))
        logger.info('Unoptimized F2=%.3lf, p=%.3lf, r=%.3lf' % (f2, p, r))
        f2, p, r = f2pr(yt, yp_opt)
        logger.info('Optimized   F2=%.3lf, p=%.3lf, r=%.3lf' % (f2, p, r))

        # Record optimized metrics.
        logs['val_F2'] = f2
        logs['val_prec'] = p
        logs['val_reca'] = r

        # Let em know you improved.
        if f2 > self.best_metric:
            logger.info('val_F2 improved from %.3lf to %.3lf %s!' % (self.best_metric, f2, u'\U0001F642'))
            self.best_metric = f2
            self.best_epoch = epoch
        else:
            logger.info('Last improvement %d epochs ago %s. Maybe next time!' %
                        ((epoch - self.best_epoch), u'\U0001F625'))


class HistoryPlotCB(Callback):
    '''Plots all of the metrics in a single figure and saves to the given file name. Plots the same metric's validation and training values on the same subplot for easy comparison and overfit monitoring.'''

    def __init__(self, file_name=None):
        super(Callback, self).__init__()
        self.logs = {}
        self.file_name = file_name

    def on_train_begin(self, logs={}):
        self.logs = {}

    def on_epoch_end(self, epoch, logs={}):
        if self.file_name is not None:
            import matplotlib
            matplotlib.use('agg')
        import matplotlib.pyplot as plt
        if len(self.logs) == 0:
            self.logs = {key: [] for key in logs.keys()}
        for key, val in logs.items():
            self.logs[key].append(val)
        keys = sorted([k for k in self.logs.keys() if not k.startswith('val')])
        nb_metrics = len(keys)
        keys = iter(keys)
        nb_col = 6
        nb_row = int(ceil(nb_metrics * 1.0 / nb_col))
        fig, axs = plt.subplots(nb_row, nb_col, figsize=(min(nb_col * 3, 12), 3 * nb_row))
        for idx, ax in enumerate(fig.axes):
            if idx >= nb_metrics:
                ax.axis('off')
                continue
            key = next(keys)
            ax.set_title(key)
            ax.plot(self.logs[key], label='TR')
            val_key = 'val_%s' % key
            if val_key in self.logs:
                ax.plot(self.logs[val_key], label='VL')
            ax.legend()
        plt.suptitle('Epoch %d: %s' % (epoch, ctime()), y=1.10)
        plt.tight_layout(pad=0.8, w_pad=0.8, h_pad=1.0)
        if self.file_name is not None:
            plt.savefig(self.file_name)
            plt.close()
        else:
            plt.show()
            plt.close()


def name_layers(model):
    '''Gives the layers sensible names in alphabetical order for use with Tensorboard.'''
    model.layers[0].name = 'name_layers'
    import pdb
    pdb.set_trace()
