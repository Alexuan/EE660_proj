import argparse

import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.semi_supervised import LabelPropagation, LabelSpreading

from dataload import VowelDataset


def train_eval_worker(model, label_percent, 
                      train_feat, train_label, train_orig_label, 
                      eval_feat, eval_label, mode='SL'):
    if mode == 'SL':
        num_labeled = int(label_percent / 100 * train_feat.shape[0]) - 1
        num_labeled = num_labeled if num_labeled < train_feat.shape[0] - 100 else train_feat.shape[0] - 100
        feat_train = train_feat[:num_labeled, :]
        label_train = train_orig_label[:num_labeled]
        model.fit(feat_train, label_train)
    elif mode == 'Semi-SL':
        feat_train = train_feat[:-100, :]
        label_train = train_label[:-100]
        model.fit(feat_train, label_train)
    else:
        raise ValueError('Not Implemented')
    return model.score(eval_feat, eval_label)



def main(args):
    ###########################################################################
    # Dataset
    dataset_trs = VowelDataset(data_dir='data/SSC_20labeled',)
    dataset_tra20 = VowelDataset(data_dir='data/SSC_20labeled',
                                 label_percent=20,)
    dataset_tra30 = VowelDataset(data_dir='data/SSC_30labeled',
                                 label_percent=30,)
    dataset_tra40 = VowelDataset(data_dir='data/SSC_40labeled',
                                 label_percent=40,)
    dataset_tst = VowelDataset(data_dir='data/SSC_20labeled', is_train=False)
    ###########################################################################
    # Models
    clf_list = []
    ### Baselines
    # TODO
    ### Supervised Learning
    if 'perceptron' in args.model_type:
        param_list = [10**i for i in range(-3, 2)]
        clf_list.extend([Perceptron(tol=tol, random_state=0) for tol in param_list])
    if 'svm' in args.model_type:
        param_list = [10**i for i in range(-3, 2)]
        clf_list.extend([SVC(kernel='linear', C=C) for C in param_list])
        clf_list.extend([SVC(kernel='rbf', C=C) for C in param_list])
    if 'mlp' in args.model_type:
        param_list = [1000, 2000, 5000]
        clf_list.extend([MLPClassifier(max_iter=max_iter) for max_iter in param_list])
    if 'random_forest' in args.model_type:
        clf_list.extend([RandomForestClassifier(max_depth=12, n_estimators=40)])
    if 'adaboost' in args.model_type:
        clf_list.extend([AdaBoostClassifier(n_estimators=2000, learning_rate = 0.5)])
    ### Semi-Supervised Learning
    if 'semi_prop' in args.model_type:
        clf_list.extend([LabelPropagation()])
    if 'semi_spread' in args.model_type:
        clf_list.extend([LabelSpreading()])
    ### Proposed Method
    
    ###########################################################################
    # Train & Test
    percentage_list = [100, 20, 30, 40]
    dataset_list = [dataset_trs, dataset_tra20, dataset_tra30, dataset_tra40]

    for percentage, data in zip(percentage_list, dataset_list):
        ### train
        acc_mean_list = []
        acc_std_list = []
        for clf in clf_list:
            acc_list_sub = []
            for i in range(10):
                acc = train_eval_worker(model=clf,
                                        label_percent=percentage,
                                        train_feat=data.feat_list[i],
                                        train_label=data.label_list[i],
                                        train_orig_label=data.orig_label_list[i],
                                        eval_feat=data.feat_list[i][-100:,:],
                                        eval_label=data.orig_label_list[i][-100:],
                                        mode='Semi-SL')
                acc_list_sub.append(acc)
            acc_sub_mean = np.mean(np.array(acc_list_sub))
            acc_sub_std = np.std(np.array(acc_list_sub))
            acc_mean_list.append(acc_sub_mean)
            acc_std_list.append(acc_sub_std)
        acc_max_index = np.argmax(np.array(acc_mean_list))
        ### test
        clf_sele = clf_list[acc_max_index]
        acc_list_test = []
        for i in range(10):
            acc = train_eval_worker(model=clf_sele,
                                    label_percent=percentage,
                                    train_feat=data.feat_list[i],
                                    train_label=data.label_list[i],
                                    train_orig_label=data.orig_label_list[i],
                                    eval_feat=dataset_tst.feat_list[i],
                                    eval_label=dataset_tst.orig_label_list[i],
                                    mode='Semi-SL')
            acc_list_test.append(acc)
        acc_test_mean = np.mean(np.array(acc_list_test))
        acc_test_std = np.std(np.array(acc_list_test))
        print('============= percentage: {} ============'.format(percentage))
        print('============= model: {} ============'.format(clf_sele))
        print('============= acc mean: {} ============'.format(acc_test_mean))
        print('============= acc std: {} ============'.format(acc_test_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EE660 Project Team7')
    parser.add_argument('--model_type', type=str, default='', 
                        help='type of model')
    args = parser.parse_args()
    main(args)