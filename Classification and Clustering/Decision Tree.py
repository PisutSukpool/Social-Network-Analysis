#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import tree
from sklearn import datasets
wine_data = datasets.load_wine()

lst_target = wine_data.target.tolist()
lst_data = wine_data.data.tolist()
X = pd.DataFrame(lst_data, columns =['Alchol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium', 'Total_phenols', 'Falvanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280', 'Proline'])
y = pd.DataFrame(lst_target, columns = ['Cultivator'])    
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.3, random_state = 1)

clf = DecisionTreeClassifier(criterion = 'entropy', random_state = 1, max_depth=2 ) #max_depth=3, min_samples_leaf=5)
clf.fit(X_train, y_train)
feature_2 = ['Alchol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium', 'Total_phenols', 'Falvanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280', 'Proline']
target = ['0','1','2']


# In[51]:


from sklearn.externals.six import StringIO 
from IPython.display import Image 
from sklearn.tree import export_graphviz
import pydotplus
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
 feature_names=feature_2, 
 class_names=target,
 filled=True, rounded=True,
 special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue()) 
Image(graph.create_png())


# In[52]:


y_pred = clf.predict(X_test)

print('Accuracy of Decision Tree classifier on training set: {:.4f}'
     .format(clf.score(X_train, y_train)*100))
print('Accuracy of Decision Tree classifier on test set: {:.4f}'
     .format(clf.score(X_test, y_test)*100))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

