#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,mean_squared_error, r2_score, homogeneity_score

from sklearn import datasets
wine_data = datasets.load_wine()
lst_target = wine_data.target.tolist()
lst_data = wine_data.data.tolist()
X = pd.DataFrame(lst_data, columns =['Alchol', 'Malic_Acid', 'Ash', 'Alcalinity_of_Ash', 'Magnesium', 'Total_phenols', 'Falvanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'OD280', 'Proline'])
y = pd.DataFrame(lst_target, columns = ['Cultivator'])    

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit to all data
scaler.fit(X)
X_scaled = scaler.transform(X)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X_scaled)
principalDf = pd.DataFrame(data = principalComponents
 , columns = ['principal component 1', 'principal component 2'])
principalDf.head()
#Concatenating DataFrame along axis = 1. finalDf is the final DataFrame before plotting the data.
finalDf = pd.concat([principalDf, y], axis = 1)
finalDf.head()


# In[25]:


fig = plt.figure(figsize = (4,4))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
Cultivators = [0, 1, 2]
colors = ['r', 'g', 'b']
for Cultivator, color in zip(Cultivators,colors):
    indicesToKeep = finalDf['Cultivator'] == Cultivator
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 10)
ax.legend(Cultivators)
ax.grid()


# In[20]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(principalDf)
y_kmeans = kmeans.predict(principalDf)
plt.scatter(principalDf.iloc[:, 0], principalDf.iloc[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[21]:


labels_true = finalDf['Cultivator'].values
clustering_predicted = y_kmeans
print('Homogeneity score is {:.4f}'.format(homogeneity_score(labels_true, clustering_predicted)))

