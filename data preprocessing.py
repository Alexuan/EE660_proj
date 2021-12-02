# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:04:03 2021

@author: Jionghao Fang
"""

import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.semi_supervised import LabelPropagation,LabelSpreading
# from wrapper import SKTSVM




'''read data'''

'''read all-labeled data(vowel-10-1trs)'''
df = pd.read_table("data/SSC_20labeled/vowel/vowel-10-1trs.dat")
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
'''read 20% labeled data(vowel-10-1tra)'''
df = pd.read_table("data/SSC_20labeled/vowel/vowel-10-1tra.dat")
df0 = df.iloc[17:,0]
df00 = df0.tolist()

#20% labeled data
n = int(len(df00) * 0.2)
ssc_20 = df00[0:n]
data_20 = []
for i in range(len(ssc_20)):
    data_20.append([])
    l = ssc_20[i].split(',')
    for j in range(len(l)):
        data_20[i].append(float(l[j]))
feature_20 = np.array(data_20)
feature_20 = feature_20[:,:-1]
label_20 = np.array(data_20)[:,-1]

#80% unlabeled data
ssc_20_unlabeled = df00[n:]
feature = []
for i in range(len(ssc_20_unlabeled)):
    feature.append(ssc_20_unlabeled[i][:-11])

feature_80 = []
for i in range(len(feature)):
    feature_80.append([])
    l = feature[i].split(',')
    for j in range(len(l)):
        feature_80[i].append(float(l[j]))
feature_80 = np.array(feature_80)

#20% test data
df = pd.read_table("data/SSC_20labeled/vowel/vowel-10-1tst.dat")
df0 = df.iloc[17:,0]
df00 = df0.tolist()
test = []
for i in range(len(df00)):
    test.append([])
    l = df00[i].split(',')
    for j in range(len(l)):
        test[i].append(float(l[j]))
feature_test = np.array(data)[:, :-1]
label_test = np.array(data)[:,-1]
        







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








'''baseline'''

'''majority voting'''
def majority_voting(label, test):
    major = max(label, key = label.count)
    label_pred = []
    for i in range(len(test)):
        label_pred.append(major)
    return label_pred
# #try 100% labeled data
# label_all = label_all.tolist()
# feature_all = feature_all.tolist()
# y_pred_100 = majority_voting(label_all, feature_all)
# mse_100 = mse(label_all,y_pred_100)

# #try 20% labeled data
# label_20 = label_20.tolist()
# feature_20 = feature_20.tolist()
# y_pred_20 = majority_voting(label_20, feature_20)
# mse_20 = mse(label_20, y_pred_20)

# #try test data
# label_test = label_test.tolist()
# feature_test = feature_test.tolist()
# y_pred_test = majority_voting(label_all, feature_test)
# mse_test = mse(label_test, y_pred_test)

'''random_choice'''
def random_choice(label, test):
    y = label
    pred_label = []
    for i in range(len(test)):
        pred_label.append(random.choice(y))
    return pred_label

# #try 100% labeled data
# label_all = label_all.tolist()
# feature_all = feature_all.tolist()
# y_pred_100 = random_choice(label_all, feature_all)
# mse_100 = mse(label_all,y_pred_100)

# #try 20% labeled data
# label_20 = label_20.tolist()
# feature_20 = feature_20.tolist()
# y_pred_20 = random_choice(label_20, feature_20)
# mse_20 = mse(label_20, y_pred_20)

# #try test data
# label_test = label_test.tolist()
# feature_test = feature_test.tolist()
# y_pred_test = random_choice(label_all, feature_test)
# mse_test = mse(label_test, y_pred_test)










'''base models'''

'''Perceptron'''
def perceptron(fea_in, label_in, fea_out, label_out):
    perceptron = Perceptron(tol=1e-3, random_state=0)
    perceptron.fit(fea_in, label_in)
    y_pred = perceptron.predict(fea_out)
    acc = perceptron.score(fea_out, label_out)
    return y_pred, acc
# #try 100% labeled data
# y_pred_100, acc_100 = perceptron(feature_all, label_all, feature_all)
# mse_100 = mse(label_all, y_pred_100)
# #try 20% labeled data
# y_pred_20, acc_20 = perceptron(feature_20,label_20, feature_20, label_20)
# mse_20 = mse(label_20, y_pred_20)        
# #try test data(100%)
# y_pred_test_100, acc_test_100 = perceptron(feature_all,label_all, feature_test, label_test)
# mse_test_100 = mse(label_test, y_pred_test_100)   
# #try test data(20%)
# y_pred_test_20, acc_test_20 = perceptron(feature_20,label_20, feature_test, label_test)
# mse_test_20 = mse(label_test, y_pred_test_20)     
   
'''SVC(kernel = rbf)'''
def SVC_rbf(fea_in, label_in, fea_out, label_out):
    rbf_svc = SVC(kernel='rbf', C=0.5)
    rbf_svc.fit(fea_in, label_in)
    y_pred = rbf_svc.predict(fea_out)
    acc = rbf_svc.score(fea_out, label_out)
    return y_pred, acc
# #try 100% labeled data
# y_pred_100, acc_100 = SVC_rbf(feature_all, label_all, feature_all)
# mse_100 = mse(label_all, y_pred_100)
# #try 20% labeled data
# y_pred_20, acc_20 = SVC_rbf(feature_20,label_20, feature_20, label_20)
# mse_20 = mse(label_20, y_pred_20)   
# #try test data(100%)
# y_pred_test_100, acc_test_100 = SVC_rbf(feature_all,label_all, feature_test, label_test)
# mse_test_100 = mse(label_test, y_pred_test_100)   
# #try test data(20%)
# y_pred_test_20, acc_test_20 = SVC_rbf(feature_20,label_20, feature_test, label_test)
# mse_test_20 = mse(label_test, y_pred_test_20)      

'''MLP'''
def mlp(fea_in, label_in, fea_out, label_out):
    mlp = MLPClassifier(max_iter=2000).fit(fea_in, label_in)
    y_pred = mlp.predict(fea_out)
    acc = mlp.score(fea_out, label_out)
    return y_pred, acc
# #try 100% labeled data
# y_pred_100, acc_100 = mlp(feature_all, label_all, feature_all)
# mse_100 = mse(label_all, y_pred_100)
# #try 20% labeled data
# y_pred_20, acc_20 = mlp(feature_20,label_20, feature_20, label_20)
# mse_20 = mse(label_20, y_pred_20)   
# #try test data(100%)
# y_pred_test_100, acc_test_100 = mlp(feature_all,label_all, feature_test, label_test)
# mse_test_100 = mse(label_test, y_pred_test_100)   
# #try test data(20%)
# y_pred_test_20, acc_test_20 = mlp(feature_20,label_20, feature_test, label_test)
# mse_test_20 = mse(label_test, y_pred_test_20)   
     
'''random forest'''
def random_forest(fea_in, label_in, fea_out, label_out):
    rf = RandomForestClassifier(max_depth=12, n_estimators=40)
    rf.fit(fea_in, label_in)
    y_pred = rf.predict(fea_out)
    acc = rf.score(fea_out, label_out)
    return y_pred, acc
# #try 100% labeled data
# y_pred_100, acc_100 = random_forest(feature_all, label_all, feature_all,label_all)
# mse_100 = mse(label_all, y_pred_100)
# #try 20% labeled data
# y_pred_20, acc_20 = random_forest(feature_20,label_20, feature_20, label_20)
# mse_20 = mse(label_20, y_pred_20)   
# #try test data(100%)
# y_pred_test_100, acc_test_100 = random_forest(feature_all,label_all, feature_test, label_test)
# mse_test_100 = mse(label_test, y_pred_test_100)   
# #try test data(20%)
# y_pred_test_20, acc_test_20 = random_forest(feature_20,label_20, feature_test, label_test)
# mse_test_20 = mse(label_test, y_pred_test_20)   

'''adaboost'''
def adaboost(fea_in, label_in, fea_out, label_out):
    ada = AdaBoostClassifier(n_estimators=2000, learning_rate = 0.5)
    ada.fit(fea_in, label_in)
    y_pred = ada.predict(fea_out)
    acc = ada.score(fea_out, label_out)
    return y_pred, acc
#try 100% labeled data
# y_pred_100, acc_100 = adaboost(feature_all, label_all, feature_all, label_all)
# mse_100 = mse(label_all, y_pred_100)
# #try 20% labeled data
# y_pred_20, acc_20 = adaboost(feature_20,label_20, feature_20, label_20)
# mse_20 = mse(label_20, y_pred_20)   
# #try test data(100%)
# y_pred_test_100, acc_test_100 = adaboost(feature_all,label_all, feature_test, label_test)
# mse_test_100 = mse(label_test, y_pred_test_100)   
# #try test data(20%)
# y_pred_test_20, acc_test_20 = adaboost(feature_20,label_20, feature_test, label_test)
# mse_test_20 = mse(label_test, y_pred_test_20)   











'''Semi-supervised learning'''

'''label propagating'''
def semi_prop(fea_l, label_l, fea_u, fea_test, label_test):
    label_u = np.zeros(len(fea_u)) -1
    prop_label = label_l.tolist() + label_u.tolist()
    prop_fea = np.vstack((fea_l, fea_u))
    label_prop_model = LabelPropagation()
    label_prop_model.fit(prop_fea, prop_label)
    acc = label_prop_model.score(fea_test, label_test)
    return acc
# #try test data(20%)
# acc_test_20 = semi_prop(feature_20,label_20, feature_80, feature_test, label_test)



'''label spreading'''
def semi_spread(fea_l, label_l, fea_u, fea_test, label_test):
    label_u = np.zeros(len(fea_u)) -1
    spread_label = label_l.tolist() + label_u.tolist()
    spread_fea = np.vstack((fea_l, fea_u))
    spread = LabelSpreading()
    spread.fit(spread_fea, spread_label)
    acc = spread.score(fea_test, label_test)
    return acc
# #try test data(20%)
# acc_test_20 = semi_spread(feature_20,label_20, feature_80, feature_test, label_test)



'''S3VM'''
'''assign unlabeled data with the label -1'''
# def S3VM(fea_l, label_l, fea_u, fea_test, label_test):
#     label_u = np.zeros(len(fea_u)) -1
#     SSL_label = label_l.tolist() + label_u.tolist()
#     SSL_feature = np.vstack((fea_l, fea_u))
#     S3VM = SKTSVM(kernel = 'linear', lamU = 1.0)
#     S3VM.fit(SSL_feature, SSL_label) 
#     acc = S3VM.score(fea_test, label_test)
#     return acc
# #try test data(20%)
# acc_test_20 = semi_spread(feature_20, label_20, feature_80, feature_test, label_test)



'''K-means'''
'''reference:https://blog.csdn.net/vivian_ll/article/details/103494042'''
def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))

def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:,-1])
    for i in label_list:
        L_i = L[(L[:,-1])==i]
        cent_i = np.mean(L_i,0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)

def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:,:-1],U))#合并L和U
    label_list = np.unique(L[:,-1])
    k = len(label_list)           #L中类别个数
    m = np.shape(dataSet)[0]
    
    clusterAssment = np.zeros(m)#初始化样本的分配                             
    centroids = initial_centriod(L)#确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):#将每个样本分配给最近的聚类中心
            minDist = np.inf; minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    minDist = distJI; minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment 

def accuracy(true_label, pred_label):
    correct = 0
    for i in range(len(true_label)):
        if true_label[i] == pred_label[i]:
            correct += 1
    return correct / len(true_label)

L =  np.array(data_20)
U = feature_80
pred_label = semi_kMeans(L,U)
acc = accuracy(label_all, pred_label)

    