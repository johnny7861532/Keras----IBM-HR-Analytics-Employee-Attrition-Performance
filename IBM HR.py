#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:44:52 2017

@author: johnnyhsieh
"""
import pandas as pd
import numpy as np

dataset = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
#Attrition means if employees leaving the company
#check if our data contain any nan row
dataset.isnull().sum()
x_train = dataset.drop(labels = ['Attrition','Over18','StandardHours'],axis = 1)
#check if our predict values are imbalance
dataset['Attrition'].value_counts()
y_train = dataset['Attrition']


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

le = LabelEncoder()
x_train['BusinessTravel'] = le.fit_transform(x_train['BusinessTravel'])
x_train['Department'] = le.fit_transform(x_train['Department'])
x_train['EducationField'] = le.fit_transform(x_train['EducationField'])
x_train['Gender'] = le.fit_transform(x_train['Gender'])
x_train['JobRole'] = le.fit_transform(x_train['JobRole'])
x_train['MaritalStatus'] = le.fit_transform(x_train['MaritalStatus'])
x_train['OverTime'] = le.fit_transform(x_train['OverTime'])
y_train = le.fit_transform(y_train)

from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x_train, y_train)
# display the relative importance of each attribute
feature_importance = model.feature_importances_
ohe = OneHotEncoder(categorical_features =[1,3,6,10,14,16,20])
x_train = ohe.fit_transform(x_train).toarray()
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_train,y_train
                                                 ,test_size = 0.1
                                                 , random_state = 0)

import keras
from keras.layers import Dense,Dropout
from keras.models import Sequential,optimizers
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

adam = optimizers.Adam(lr = 0.0001)
Classifier = Sequential()
Classifier.add(Dense(units = 1024,activation = 'relu'
                     ,kernel_initializer="uniform", input_dim = 53))
Classifier.add(Dropout(0.6))
Classifier.add(Dense(units = 1024, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dropout(0.6))
Classifier.add(Dense(units = 512, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dropout(0.6))
Classifier.add(Dense(units = 512, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dropout(0.6))
Classifier.add(Dense(units = 256, activation = 'relu'
                     ,kernel_initializer="uniform"))
Classifier.add(Dropout(0.4))
Classifier.add(Dense(units = 128,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dropout(0.4))
Classifier.add(Dense(units = 64,activation = 'relu'
                     ,kernel_initializer = 'uniform'))
Classifier.add(Dense(units = 1,activation = 'sigmoid'
                     ,kernel_initializer="uniform"))
Classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy'
                   ,metrics = ['accuracy'])
ES = EarlyStopping(monitor='val_loss', patience=10
                           , verbose=0, mode='auto')
weight = class_weight.compute_class_weight('balanced'
                                           ,np.unique(y_train),y_train)
Classifier.fit(x_train,y_train,batch_size = 42, epochs = 100,callbacks = [ES]
,validation_data= (x_test,y_test),class_weight = weight
,shuffle=True)
predict = Classifier.predict(x_train)
predict = (predict > 0.5)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
cm = confusion_matrix(np.float32(y_train), predict)
plt.figure(figsize = (8,8))
sn.heatmap(cm, annot=True)
plt.show()

