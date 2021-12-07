import os
import copy
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt

from evaluation import heatmap, annotate_heatmap

"""
@relation vowel
@attribute TT integer [0, 1]
@attribute SpeakerNumber integer [0, 14]
@attribute Sex integer [0, 1]
@attribute F0 real [-5.211, -0.941]
@attribute F1 real [-1.274, 5.074]
@attribute F2 real [-2.487, 1.431]
@attribute F3 real [-1.409, 2.377]
@attribute F4 real [-2.127, 1.831]
@attribute F5 real [-0.836, 2.327]
@attribute F6 real [-1.537, 1.403]
@attribute F7 real [-1.293, 2.039]
@attribute F8 real [-1.613, 1.309]
@attribute F9 real [-1.68, 1.396]
@attribute Class {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, unlabeled}
@inputs TT, SpeakerNumber, Sex, F0, F1, F2, F3, F4, F5, F6, F7, F8, F9
@outputs Class
"""


class VowelDataset:
    def __init__(self, data_dir, is_train=True, num_fold=10, label_percent=100, is_norm=False):

        assert os.path.isdir(data_dir)

        if is_train == True:
            suffix = 'trs'
        elif is_train == False:
            suffix = 'tst'
        else:
            raise ValueError('Not Implemented.')

        self.feat_list = []
        self.label_list = []
        self.orig_label_list = []
        for i in range(num_fold):
            idx = i + 1
            file_dir = os.path.join(data_dir, 'vowel', 'vowel-{}-{}{}.dat'.format(num_fold, idx, suffix))
            feat, label = self._parse_data(file_dir)
            orig_label = copy.deepcopy(label)
            if is_norm:
                feat = self._norm_data(feat)
            if is_train:
                num_labeled = int(label_percent / 100 * feat.shape[0]) - 1
                label[num_labeled:] = -1.
            self.feat_list.append(feat)
            self.label_list.append(label)
            self.orig_label_list.append(orig_label)


    def _parse_data(self, file_dir):
        # TODO: need nicer method, hard code now
        df = pd.read_table(file_dir)
        df0 = df.iloc[17:,0]
        df00 = df0.tolist()
        data = []
        for i in range(len(df00)):
            data.append([])
            l = df00[i].split(',')
            for j in range(len(l)):
                data[i].append(float(l[j]))
        feature_all = np.array(data)[:, :-1]
        label_all = np.array(data)[:,-1]
        return feature_all, label_all


    def _norm_data(self, feat):
        # TODO
        return feat


    def _preprocessing(self):
        '''data preprocessing'''

        '''standardize'''
        def standardscalar(X):   
            standardScaler = StandardScaler()
            standardScaler.fit(X)
            return standardScaler.transform(X)

        '''PCA'''
        def PCA_Data(X, n=5):
            pca = PCA(n_components=n)
            pca.fit(X)
            X_reduction = pca.transform(X)
            return X_reduction

        # '''Smote'''
        # def smote(X, y):
        #     sm = SMOTE(random_state=42)
        #     return sm.fit_resample(X, y)


if __name__ == '__main__':
    dataset_tst = VowelDataset(data_dir='data/SSC_20labeled', is_train=False)
    
    feat_all = []
    for i in range(10):
        feat_all.extend(dataset_tst.feat_list[i])
    feat_all = np.stack(feat_all, axis=0)

    attr_list = ['TT',
                 'SpeakerID',
                 'Sex',
                 'F0',
                 'F1',
                 'F2',
                 'F3',
                 'F4',
                 'F5',
                 'F6',
                 'F7',
                 'F8',
                 'F9',]

    Pearson_mat = np.corrcoef(feat_all.T)
    fig, ax = plt.subplots()

    im, cbar = heatmap(Pearson_mat, attr_list, attr_list, ax=ax,
                       cmap="YlGn",)
    # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    fig.tight_layout()
    fig.savefig('pearson_mat.pdf')
