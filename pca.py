import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv


def extract_features():
    """
    To extract the column headers so we can map PC components to relevant factors
    """
    # data = pd.read_csv("Determinants (std).csv", index_col=0)
    data = pd.read_csv("data/Determinants_mn.csv", index_col=0)
    # columns = [x for x in enumerate(data.columns, 1)]
    return list(data.columns)

def extract_towns():
    """
    Extract the town names, so that health score can have meaning
    """
    data = pd.read_csv("data/Determinants_mn.csv", index_col=0)
    towns = list(data.index)
    return towns

def calc_pca(data):
    """
    Method to calculate factor scores
    """
    determinant_data = np.array(data)

    #find the number of variables
    num_var = determinant_data.shape[1]

    #calc is the choose the factor loadings
    calc = np.abs(np.sqrt(1/num_var))

    #cov_mat is the covariance matrix of the data
    cov_mat = np.cov(determinant_data.T)

    #perform PCA on the covariance matrix
    pca = PCA()
    transformed_data = pca.fit(cov_mat).transform(cov_mat)

    #proportion of variance present in each component
    eig = pca.explained_variance_
    n = len(np.where(eig>1)[0])
    transformed_data = transformed_data[:n,:n]

    #weights assigned to each pc
    weights = pca.explained_variance_ratio_/np.sum(pca.explained_variance_ratio_)

    #choose only the pc's who satisfies the constraint
    var_load = pca.components_
    var_load[var_load > calc] = 0
    
    
    #print table of indicator variable loadings
    load_table = []
    for i in range(n):
        load_table.append(var_load[:, i]) 

    features = extract_features()
    loadings = zip(features, *load_table)
    with open("output/loadings_1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(loadings)
    
    #scores assigned to each component
    factor_scores = weights @ var_load

    #calculate the health score for every town
    health_status = np.array([determinant_data @ factor_scores]).T
    health_status = MinMaxScaler().fit_transform(health_status)
    health_status = health_status.flatten()
    
    return [round(x,2) for x in health_status]

def main():
    data = pd.read_csv('data/determinant_data_mn.csv', index_col=0)
    scores = calc_pca(data)
    towns = extract_towns()

    #assign health scores
    health_scores = sorted(zip(towns, scores), key = lambda x: x[1])
    print(health_scores)

    with open("output/pca_1_mn.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(health_scores)

if __name__ == '__main__':
    features = extract_features()
    print(features)
    # main()