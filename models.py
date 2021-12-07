import numpy as np
from sklearn.metrics import f1_score
import random


class MajorityVoting():
    def fit(self, X, y):
        """
        X: feature (ignored, just for API consistency)
        y: ground truth label
        """
        counts = np.bincount(y.astype(int))
        self.major = np.argmax(counts)

    def score(self, X, y):
        """
        X: feature (ignored, just for API consistency)
        y: ground truth label
        """
        pred = np.ones_like(y) * self.major
        f1_mic = f1_score(y, pred, average='micro')
        f1_mac = f1_score(y, pred, average='macro')
        return f1_mic



class RandomChoice():
    def fit(self, X, y):
        """
        X: feature (ignored, just for API consistency)
        y: ground truth label
        """
        pass

    def score(self, X, y):
        """
        X: feature (ignored, just for API consistency)
        y: ground truth label
        """
        pred = np.random.choice(11, y.shape[0])
        f1_mic = f1_score(y, pred, average='micro')
        f1_mac = f1_score(y, pred, average='macro')
        return f1_mic


