# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 08:01:16 2019

@author: Virendra
"""
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

objects = []
with (open("Data_preprocessing/PAMAP2_Dataset/Protocol/activity6.pkl", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

objects1 = []
with (open("Data_preprocessing/PAMAP2_Dataset/Protocol/activity6.pkl", "rb")) as openfile:
    while True:
        try:
            objects1.append(pickle.load(openfile))
        except EOFError:
            break

features = objects[0]['data'].drop(['imu1temp','imu2temp','imu3temp','activityid'],axis=1)
labels = np.array(objects[0]['target'])
labels = labels.reshape((len(labels),1))

features1 = objects1[0]['data']
labels1 = np.array(objects1[0]['target'])
labels1 = labels1.reshape((len(labels1),1))

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.25, random_state=0)
x1_train, x1_test, y1_train, y1_test = train_test_split(features1, labels1, test_size=0.25, random_state=0)      

gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
print("Naive Bayes Accuracy:",metrics.accuracy_score(y_test, y_pred))
openfile.close()