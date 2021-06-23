#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[31]:


#SKlearn prokject 

from  sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
iris = pd.read_csv('http://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')

y = iris['species'].astype('category').cat.codes
x = iris.iloc[:, :4]
x = MinMaxScaler().fit_transform(x)#no need to scale y as it is a target categorical variable i.e: vals are 0,1,2 therefore no need to scale

pca = PCA(n_components = 2, random_state = 0 )
pCs = pca.fit_transform(x)


knn = KNeighborsClassifier(n_neighbors = 6, algorithm = 'auto', p = 2)

results = cross_val_score(knn, pCs, y, cv = 33)
print(results.mean())


# In[ ]:




