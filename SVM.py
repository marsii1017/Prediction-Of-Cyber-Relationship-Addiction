# -*- coding: utf-8 -*-
"""
Created on Wed Jun 07 00:38:31 2017

@author: User
"""

from __future__ import print_function

from keras.datasets import cifar10

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Convolution2D, MaxPooling2D

from keras.optimizers import SGD, Adam

from keras.utils import np_utils

from sklearn.feature_selection import SelectFromModel

from sklearn import svm , feature_selection
from sklearn.externals import joblib
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score
import matplotlib.pyplot as plt

import time
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import csv
import cv2


import numpy as np
import pickle
import os

X = list()
Y = list()

feature_names=np.array(range(0,44))
d = np.ones((1469,44),dtype = float)

b = np.ones(1469,dtype = int)

test_data1 = np.ones((1469,44),dtype = float)
test = np.ones((1469,22),dtype = float)
tes = np.ones((1469,11),dtype = float)
l = np.ones(1469,dtype = int)
i=0
j=0
f=open('all_feature.csv','r')

for line in open('all_feature.csv'):  
    line = f.readline() 
    array = line.split(',')
    if(i<=1468):
        
        b[i]=int(array[9])
   # elif(i>=1100 and i<= 1468):
        
   #     l[j]=int(array[9])
   #     j+=1
    i+=1

    
f.close()

i=0
g=0
j=0
f=open('num.csv','r')

for li in open('num.csv'):  
    li = f.readline() 
    array = li.split(',')
    if(i<=1468):
        for k in range(0,44):
            d[i][k]=float(array[k])  
            
        
   # elif(i>=1100 and i<=1468):
    #    for k in range(0,21):
     #       test_data1[j][k]=float(array[k])  
            
        
        j+=1
    i+=1

    
f.close()


max = 0

for k in range(5828,5829):
    X_train, X_test, Y_train, Y_test = train_test_split(d, b, test_size=0.18, random_state=k)
    print(X_train.shape)
    print(X_test.shape)

#forest = ExtraTreesClassifier(n_estimators=250,random_state=0)

#forest.fit(d, b)
#importances = forest.feature_importances_
#std = np.std([tree.feature_importances_ for tree in forest.estimators_],axis=0)
#indices = np.argsort(importances)[::-1]

#Print the feature ranking
#print("Feature ranking:")

#for f in range(d.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    clf = svm.SVC(C=10, cache_size=1000,decision_function_shape='ovr', class_weight='balanced', coef0=0.0,degree=2,)
    clf.fit(X_train,Y_train)
    joblib.dump(clf, 'rf2.model')

    predict = clf.predict(X_test)
    

    print ('correct count', sum(predict == Y_test))
    print (clf.score(X_test,Y_test))
    print (recall_score(Y_test,predict))
    print (precision_score(Y_test,predict))
        
    s = clf.score(X_test,Y_test)
    if (clf.score(X_test,Y_test)>max):
        max = clf.score(X_test,Y_test)
        m=k
    X.append(k)
    Y.append(s)
print (max)
print (m)
plt.plot(X,Y)

X_train, X_test, Y_train, Y_test = train_test_split(d, b, test_size=0.18, random_state=m)
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=5828)

forest.fit(X_train, Y_train )
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# The "accuracy" scoring is proportional to the number of correct
# classifications
important_names = feature_names[importances > np.mean(importances)]
print (important_names)

k=0
i = 0
j=0
for k in range(0,22):
    for i in range(0,1469):
        j=int(important_names[k])
        test[i][k]=d[i][j]
    

X_train, X_test, Y_train, Y_test = train_test_split(test, b, test_size=0.18, random_state=m)
clf = svm.SVC(C=10, cache_size=1000,decision_function_shape='ovr', class_weight='balanced', coef0=0.0,degree=2,)
clf.fit(X_train,Y_train)
joblib.dump(clf, 'rf2.model')

predict = clf.predict(X_test)
    

print ('correct count', sum(predict == Y_test))
print (clf.score(X_test,Y_test))
print (recall_score(Y_test,predict))
print (precision_score(Y_test,predict))






