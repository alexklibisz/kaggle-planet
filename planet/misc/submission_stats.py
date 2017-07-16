# Simple CLI program to show stats about a submission CSV.
# Stats for test single submission file:
# - Class proportions (how many negatives, positives).
# Additional for training submission file:
# - F2, precision, recall for each tag.
from termcolor import colored
import argparse
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import sys
sys.path.append('.')
from planet.utils.data_utils import TAGS, tagstr_to_binary, f2pr
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    logger = logging.getLogger(__name__)
    ap = argparse.ArgumentParser(description='Model runner.')
    ap.add_argument('-g', '--ground-truth', help='path to ground truth', default='data/train_v2.csv')
    ap.add_argument('-s', '--submission', help='path to submission', required=True)
    args = vars(ap.parse_args())

    # Read CSVs.
    dfgt = pd.read_csv(args['ground_truth'])
    dfsb = pd.read_csv(args['submission'])
    TRAINING = dfgt.shape[0] == dfsb.shape[0]

    # Convert tags to vectors.
    yt = np.array([tagstr_to_binary(ts) for ts in dfgt['tags']])
    yp = np.array([tagstr_to_binary(ts) for ts in dfsb['tags']])

    # Compute class metrics: proportion positive, f2, precision, recall.
    for i, tag in enumerate(TAGS):
        pos_yt = np.sum(yt[:, i])
        pos_yp = np.sum(yp[:, i])
        diff = abs(pos_yt - pos_yp)
        s = '%-20s pos_yt=%05d pos_yp=%05d (%s%05d)' % \
            (tag, pos_yt, pos_yp, '+' if pos_yp > pos_yt else '-', diff)

        if TRAINING:
            f2, p, r = f2pr(yt[:, i], yp[:, i])
            s += ' f2=%.3lf prec=%.3lf reca=%.3lf' % (f2, p, r)

            if f2 > 0.9:
                s = colored(s, 'green')
            elif f2 > 0.8:
                s = colored(s, 'yellow')
            elif f2 > 0.7:
                s = colored(s, 'magenta')
            else:
                s = colored(s, 'red')

        logger.info(s)

    if TRAINING:
        logger.info('Key: %s %s %s %s' % (colored('> 0.9', 'green'), colored('> 0.8', 'yellow'),
                                          colored('> 0.7', 'magenta'), colored('< 0.7', 'red')))

        f2, p, r = f2pr(yt, yp)
        logger.info('Aggregate: f2=%.3lf prec=%.3lf reca=%.3lf' % (f2, p, r))

        # Scatter plot metrics vs. their proportion of positives.
        fig, _ = plt.subplots(1, 3, figsize=(10, 5))
        [axf2, axprec, axreca] = fig.axes
        axf2.set_ylabel('F2')
        axf2.set_xlabel('pos_yt')
        axprec.set_ylabel('prec')
        axprec.set_xlabel('pos_yt')
        axreca.set_ylabel('reca')
        axreca.set_xlabel('pos_yt')
        for i, tag in enumerate(TAGS):
            pos_yt = np.sum(yt[:, i])
            f2, p, r = f2pr(yt[:, i], yp[:, i])
            axf2.scatter([pos_yt], [f2])
            axprec.scatter([pos_yt], [p])
            axreca.scatter([pos_yt], [r])
        plt.show()
    
    
