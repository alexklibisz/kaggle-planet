from keras.callbacks import Callback, TensorBoard
from keras.engine import Layer
from math import ceil
from shutil import copyfile
from sklearn.metrics import fbeta_score, precision_score, recall_score
from time import sleep, ctime
from tqdm import tqdm
import keras.backend as K
import logging
import numpy as np
import os
import pickle as pkl

from planet.utils.data_utils import TAGS


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


class ParamStatsCB(Callback):
    '''Compute metrics about network parameters over time.'''

    def __init__(self, cpdir):
        super(Callback, self).__init__()
        self.cpdir = cpdir
        self.trainable_lidxs = []
        self.lidx_w_prev = None
        self.lidx_b_prev = None
        self.lidx_wstd = None
        self.lidx_b_std = None
        self.lidx_wratio = None
        self.lidx_b_ratio = None
        self.counter = 0

    def on_train_begin(self, logs):
        self.trainable_lidxs = [i for i, l in enumerate(self.model.layers)
                                if len(l.trainable_weights) > 0]
        self.lidx_wb_prev = {li: None for li in self.trainable_lidxs}
        self.lidx_wstats = {li: [] for li in self.trainable_lidxs}
        self.lidx_wratio = {li: [] for li in self.trainable_lidxs}

    def on_batch_end(self, batch, logs):
        '''Compute metrics and update for next iteration.'''

        if batch % 10:
            return

        lidx_wb = {li: self.model.layers[li].get_weights()[0:] for li in self.trainable_lidxs}

        # Compute comparisons against previous weights.
        # - Standard deviation of weights.
        # - Ratio of weight magnitude to the most recent update delta.
        if self.counter > 0:
            for li in self.trainable_lidxs:
                w0 = self.lidx_wb_prev[li][0]
                w1 = lidx_wb[li][0]
                self.lidx_wstats[li].append((np.mean(w1), np.var(w1)))
                self.lidx_wratio[li].append(np.mean(np.abs(w1 - w0) / (np.abs(w1) + 1e-7)))

        self.lidx_wb_prev = lidx_wb
        self.counter += 1

    def on_epoch_end(self, epoch, logs):
        '''Print metrics for this epoch and save metrics to disk.'''
        print('\n Epoch %d: mean stats for each layer:' % (epoch))
        for li in self.trainable_lidxs:
            layer = self.model.layers[li]
            means = [m for m, s in self.lidx_wstats[li][-self.counter:]]
            varcs = [s for m, s in self.lidx_wstats[li][-self.counter:]]
            print('%-30s w mean=%-8.4lf w variance=%-8.4lf' % (layer.name, np.mean(means), np.mean(varcs)))
        self.counter = 0
        payload = {'lidx_wstats': self.lidx_wstats, 'lidx_wratio': self.lidx_wratio}
        with open('%s/param_stats.pkl' % self.cpdir, 'wb') as f:
            pkl.dump(payload, f)


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

    def __init__(self, cpdir, batch_gen, batch_size, nb_steps, threshold):
        super(Callback, self).__init__()
        self.cpdir = cpdir
        self.batch_size = batch_size
        self.batch_gen = batch_gen
        self.nb_steps = nb_steps
        self.metrics = {'f2': [], 'prec': [], 'reca': []}
        self.tag_metrics = {t: {'f2': [], 'prec': [], 'reca': []} for t in TAGS}
        self.threshold = threshold
        assert os.path.exists(self.cpdir)

    def on_epoch_end(self, epoch, logs):
        '''
        1. Predictions on every image in the validation set.
        2. Evaluate the precision, recall, and F2 of each tag.
        3. Store the metrics and predictions in a pickle file in the cpdir.
        '''

        # Make predictions and store all true and predicted tags. Round predictions.
        yt = np.zeros((self.batch_size * self.nb_steps, len(TAGS)))
        yp = np.zeros((self.batch_size * self.nb_steps, len(TAGS)))
        didxs = np.zeros((self.nb_steps, self.batch_size))
        for bidx in tqdm(range(self.nb_steps)):
            ib, tb, db = next(self.batch_gen)
            yt[bidx * self.batch_size:(bidx + 1) * self.batch_size] = tb
            yp[bidx * self.batch_size:(bidx + 1) * self.batch_size] = self.model.predict(ib) > self.threshold
            didxs[bidx] = db

        # Metrics per tag.
        for tidx, tag in enumerate(TAGS):
            self.tag_metrics[tag]['f2'].append(fbeta_score(yt[:, tidx], yp[:, tidx], beta=2.))
            self.tag_metrics[tag]['prec'].append(precision_score(yt[:, tidx], yp[:, tidx]))
            self.tag_metrics[tag]['reca'].append(recall_score(yt[:, tidx], yp[:, tidx]))

        # Mean metrics across all examples.
        f2, prec, reca = 0., 0., 0.
        for i in range(yt.shape[0]):
            f2 += fbeta_score(yt[i], yp[i], beta=2.) / yt.shape[0]
            prec += precision_score(yt[i], yp[i]) / yt.shape[0]
            reca += recall_score(yt[i], yp[i]) / yt.shape[0]
        self.metrics['f2'].append(f2)
        self.metrics['prec'].append(prec)
        self.metrics['reca'].append(reca)

        # Record Keras metrics.
        logs['val_F2'] = f2
        logs['val_prec'] = prec
        logs['val_reca'] = reca

        # Save to disk.
        payload = {'metrics': self.metrics, 'tag_metrics': self.tag_metrics, 'yt': yt, 'yp': yp, 'didxs': didxs}
        p = '%s/val_data_%d.pkl' % (self.cpdir, epoch)
        with open(p, 'wb') as fp:
            pkl.dump(payload, fp)
        copyfile(p, '%s/val_data_latest.pkl' % self.cpdir)


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
