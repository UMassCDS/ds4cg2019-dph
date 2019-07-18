import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def calc_pca():
	data = pd.read_csv('determinant_data.csv', index_col=0)
	# data.set_index('Unnamed: 0', inplace = True)
	determinant_data = np.array(data)

	num_var = determinant_data.shape[1]

	calc = np.abs(np.sqrt(1/num_var))

	cov_mat = np.cov(determinant_data.T)

	pca = PCA()

	transformed_data = pca.fit(cov_mat).transform(cov_mat)

	eig = pca.explained_variance_

	n = len(np.where(eig>1)[0])

	transformed_data = transformed_data[:n,:n]

	weights = pca.explained_variance_ratio_/np.sum(pca.explained_variance_ratio_)

	var_load = pca.components_
	var_load[var_load > calc] = 0

	factor_scores = weights @ var_load

	health_status = np.array([determinant_data @ factor_scores]).T

	health_status = MinMaxScaler().fit_transform(health_status)

	print(health_status.shape)
	print(health_status)

calc_pca()