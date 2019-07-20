#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

class pc_component_scores:
    
    def __init__(self):
        
        # needs to built based on how we will use the pc component scores to compute community risks
        pass
    
    def correlation_matrix(self, x):
        """
        Computes the correlation matrix for given data values

        PARAMETERS
        ----------
        - x: the data values

        RETURNS
        =======
        - corr_mat: correlation matrix
        """
        cov_mat = np.cov(x.T)
        return cov_mat

    def compute_component_score(self, x):
        """
        Applies PCA to observational data and then computes `component scores`.
        
        PARAMETERS
        ----------
        - x (numpy array) : the correlation matrix
        
        RETURNS
        -------
        - Factor scores computed for the chosen Principal Components
        
        REFERENCE
        ---------
        [1.] https://stats.stackexchange.com/questions/102882/steps-done-in-factor-analysis-compared-to-steps-done-in-pca
        """
        pca = PCA(svd_solver='full', random_state=1)
        N, D = x.shape
        pca.fit(x)
        
        # kaiser criterion : Components with eigen_values > 1.0 should be retained
        selected_eigvalues = np.argwhere(pca.explained_variance_>1.0).flatten()
        selected_vr = pca.explained_variance_ratio_[selected_eigvalues]
        
        # second criterion : Among selected components, retain those with proportion of variance  > 10%
        mod_idxs = np.argwhere(selected_vr.flatten()>0.1)
        selected_vr = selected_vr[selected_vr>0.1]
        selected_eigvalues = selected_eigvalues[mod_idxs].flatten()

        n_pcs = len(selected_eigvalues)
        selected_components = pca.components_[:, :n_pcs]

        loadings = selected_components * np.sqrt(selected_eigvalues)
        component_scores = loadings @ np.diag(1/selected_eigvalues)

        return component_scores       
        
def extract_towns():
        """
        Extract the town names, so that health score can have meaning
        """
        data = pd.read_csv("Determinants (std).csv", index_col=0)
        towns = list(data.index)
        return towns

def extract_features():
    """
    To extract the column headers so we can map PC components to relevant factors
    """
    data = pd.read_csv("Determinants (std).csv", index_col=0)
    columns = [x for x in enumerate(data.columns)]
    return dict(columns)
    
if __name__=='__main__':
    pc_obj = pc_component_scores()
    data = pd.read_csv('determinant_data.csv', index_col=0)
    cov_mat = pc_obj.correlation_matrix(data)
    comp_scores = pc_obj.compute_component_score(cov_mat)
    print(comp_scores)
    features = extract_features()
    
    #calculate the health score for every town
    # health_status = np.array([np.array(data) @ comp_scores]).T
    # health_status = MinMaxScaler().fit_transform(health_status)
    # health_status = health_status.flatten()
    # health_status = [round(x,2) for x in health_status]
    
    # towns = extract_towns()
    # health_scores = sorted(zip(towns, health_status), key = lambda x: x[1])

    # with open("pca_3.csv", "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(health_scores)  