# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 13:04:03 2021

@author: Jionghao Fang
"""

import pandas as pd
import numpy as np
import random
'''read all-labeled data'''
# df = pd.read_table("vowel-10-1trs.dat")
df = pd.read_table("vowel-10-1tra.dat")
df0 = df.iloc[17:,0]
df00 = df0.tolist()
data = []
for i in range(len(df00)):
    data.append([])
    l = df00[i].split(',')
    for j in range(len(l)):
        data[i].append(float(l[j]))
feature = np.array(data)
label = np.array(data)

# '''normalize F0-F9 features'''
# F0_9 = data_X[:, 3:]
# mean = np.mean(F0_9, axis = 0)
'''baseline_majority_voting'''
def maj_voting(label, test):
    major = max(label, key = label.count)
    label_pred = []
    for i in range(len(test)):
        label_pred.append(major)
    return label_pred

'''baseline_random_choice'''
def rand_choice(label, test):
    prob = []
    tot = len(label)
    prob = np.zeros(10)
    for i in range(10):
        prob[i] = label.count(i) / tot
    
    pred_label = []
    pr = 0
    for j in range(len(test)):
        num = random.random()
        for m in range(10):
            for n in range(m):
                pr += prob[n]
            if num <= pr:
                pred_label.append(m)
    return pred_label

'''100% labeled data training'''

        
        
        
