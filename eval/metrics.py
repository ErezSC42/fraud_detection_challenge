import numpy as np
from constants import TRAIN_SEGMENT_COUNT
from sklearn.metrics import precision_score, recall_score, f1_score


def count_correct_anomalies(pred_, true_):
    counter = 0
    for x, y in zip(pred_, true_):
        if y == -1 and x == -1:
            counter +=1
    return counter / 10


def detection_metrics(pred_, true_):
    return {
        "detection_score": count_correct_anomalies(pred_, true_),
        "precision_score": float(f"{precision_score(pred_, true_):.3f}"),
        "recall_score": float(f"{recall_score(pred_, true_):.3f}"),
        "f1_score": float(f"{f1_score(pred_, true_):.3f}")
    }


class ScoreCounter:
    def __init__(self, n_users: int):
        self.n_users = n_users
        self.counter = 0    # max should be 1800
        self.div = (9 * 10 + 1 * 90) * self.n_users   # should be 1800

    def update(self, pred_: np.array, gt_: np.array):
        ''' make calucations '''
        for x, x_ in zip(gt_, pred_):
            if x == x_ == -1:
                self.counter += 9
            elif x == x_ == 1:
                self.counter += 1

    def calc(self):
        return self.counter, self.counter / self.div
