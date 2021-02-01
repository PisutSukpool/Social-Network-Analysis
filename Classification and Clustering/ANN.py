#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier

from sklearn import datasets
wine_data = datasets.load_wine()
lst_target = wine_data.target.tolist()
lst_data = wine_data.data.tolist()
X = pd.DataFrame(lst_data, columns =['Alchol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium', 'Total_phenols', 'Falvanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280', 'Proline'])
y = pd.DataFrame(lst_target, columns = ['Cultivator'])    
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 1)

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

NN = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
NN.fit(X_train,y_train)

NN_scaled = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
NN_scaled.fit(X_train_scaled,y_train)


y_pred_scaled = NN_scaled.predict(X_test_scaled)  
y_pred_scaled


print('Accuracy of NN classifier on training set: {:.4f}'
     .format(NN_scaled.score(X_train_scaled, y_train)*100))
print('Accuracy of NN classifier on test set: {:.4f}'
     .format(NN_scaled.score(X_test_scaled, y_test)*100))

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_pred_scaled))
print(confusion_matrix(y_test,y_pred_scaled))

