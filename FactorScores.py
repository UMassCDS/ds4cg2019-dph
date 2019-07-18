#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:08:19 2019

@author: roshanprakash
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

class HealthRiskModel:
    
    def __init__(self):
        
        # needs to built based on how we will use the factor scores to compute community risks
        pass
    
    def _compute_factor_scores(self, x):
        """
        Applies PCA to observational data and then computes `factor scores`.
        
        PARAMETERS
        ----------
        - x (numpy array) : the input data
        
        RETURNS
        -------
        - Factor scores computed for the chosen Principal Components
        
        REFERENCE
        ---------
        [1.] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5065523/
        """
        pca = PCA(svd_solver='full', random_state=1)
        N, D = x.shape
        pca.fit(x)
        
        # kaiser criterion : Components with eigen_values > 1.0 should be retained
        selected_components = np.argwhere(pca.explained_variance_>1.0).flatten()
        selected_vr = pca.explained_variance_ratio_[selected_components]
        
        # second criterion : Among selected components, retain those with proportion of variance  > 10%
        mod_idxs = np.argwhere(selected_vr.flatten()>0.1)
        selected_vr = selected_vr[selected_vr>0.1]
        selected_components = selected_components[mod_idxs].flatten()
        component_weights = selected_vr/np.sum(selected_vr)
        
        # influential variables 
        cutoff = np.abs(np.sqrt(1/D))
        influential_vars = {c : list(np.argwhere(pca.components_[c, :]>cutoff-0.1).flatten()) for c in selected_components}
        
        # compute factor scores per component
        factor_scores = np.sum(pca.components_[selected_components]*component_weights[:, np.newaxis], axis=1)
        
        return factor_scores, influential_vars
    
if __name__=='__main__':
    rml = HealthRiskModel()
    df = pd.read_csv('determinant_data.csv', index_col=0)
    factor_scores, _ = rml._compute_factor_scores(df.values)
    print(factor_scores)
        