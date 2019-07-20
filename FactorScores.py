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
from sklearn.preprocessing import MinMaxScaler
import csv
from collections import defaultdict

class HealthRiskModel:
    
    def __init__(self):
        
        # needs to built based on how we will use the factor scores to compute community risks
        pass
    
    def _compute_factor_score(self, x):
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
        influential_vars = {c:list(np.argwhere(pca.components_[c, :]>cutoff-0.1).flatten()) for c in selected_components}
        
        # compute factor score
        factor_score = np.sum(pca.components_[selected_components]*component_weights[:, np.newaxis], axis=0)
     
        return factor_score, influential_vars

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
    rml = HealthRiskModel()
    df = pd.read_csv('determinant_data.csv', index_col=0)
    factor_scores, influential_vars = rml._compute_factor_score(df.values)
    
    features = extract_features()
    pc_features = {}
    
    for key,val in influential_vars.items():
        pc_features[key] = [features[x] for x in val]
    
    #calculate the health score for every town
    health_status = np.array([np.array(df) @ factor_scores]).T
    health_status = MinMaxScaler().fit_transform(health_status)
    health_status = health_status.flatten()
    health_status = [round(x,2) for x in health_status]
    
    towns = extract_towns()
    health_scores = sorted(zip(towns, health_status), key = lambda x: x[1])

    with open("pca_2.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(health_scores)  