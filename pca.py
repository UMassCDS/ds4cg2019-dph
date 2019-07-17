import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

data = pd.read_csv('../determinant_data.csv')

data.set_index('Unnamed: 0', inplace = True)

determinant_data = np.array(data)

num_var = determinant_data.shape[1]

calc = np.abs(np.sqrt(1/num_var))

pca = PCA()

transformed_data = pca.fit(determinant_data).transform(determinant_data)

eig = pca.explained_variance_

n = len(np.where(eig>1)[0])

pca = PCA(n_components=n)

transformed_data = pca.fit(determinant_data).transform(determinant_data)

weights = pca.explained_variance_ratio_/np.sum(pca.explained_variance_ratio_)

var_load = pca.components_
var_load[var_load > calc] = 0

factor_scores = weights @ var_load

print(factor_scores.shape)
print(factor_scores)
