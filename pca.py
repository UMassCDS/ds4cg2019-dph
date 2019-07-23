import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv

def comp_pca(data):
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

    #scores assigned to each component
    factor_scores = weights @ var_load

    #calculate the health score for every town
    health_status = np.array([determinant_data @ factor_scores]).T
    health_status = MinMaxScaler().fit_transform(health_status)
    health_status = health_status.flatten()
    
    scores = [round(x,2) for x in health_status]
    
    return scores

class HealthScores():
    '''
    Get healthscores for all towns
    '''
    def __init__(self, cols_filepath, data_filepath, pca_filepath, loadings_filepath=None):
        '''
        Initialize variables
        '''
        self.read_cols = cols_filepath
        self.data = data_filepath
        self.output = pca_filepath
        self.loadings_file = loadings_filepath

        self.var_load = None
        self.n = None
        
    def extract_features(self):
        """
        To extract the column headers so we can map PC components to relevant factors
        """
        data = pd.read_csv(self.read_cols, index_col=0)
        # columns = [x for x in enumerate(data.columns, 1)]
        return list(data.columns)

    def extract_towns(self):
        """
        Extract the town names, so that health score can have meaning
        TODO: FIX FOR ALL PCA
        """
        data = pd.read_csv(self.read_cols, index_col=0)
        # TODO: clean
        # data = pd.read_csv('data/clean_determinants_std.csv', index_col=0)
        towns = list(data.index)
        return towns

    def calc_pca(self):
        """
        Method to calculate factor scores
        """
        data = pd.read_csv(self.data, index_col=0)
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
        self.n = len(np.where(eig>1)[0])
        transformed_data = transformed_data[:self.n,:self.n]

        #weights assigned to each pc
        weights = pca.explained_variance_ratio_/np.sum(pca.explained_variance_ratio_)

        #choose only the pc's who satisfies the constraint
        self.var_load = pca.components_
        self.var_load[self.var_load > calc] = 0

        #scores assigned to each component
        factor_scores = weights @ self.var_load

        #calculate the health score for every town
        health_status = np.array([determinant_data @ factor_scores]).T
        health_status = MinMaxScaler().fit_transform(health_status)
        health_status = health_status.flatten()
        
        scores = [round(x,2) for x in health_status]
        towns = self.extract_towns()
        
        #assign health scores
        health_scores = sorted(zip(towns, scores), key = lambda x: x[1])
        print(health_scores)

        with open(self.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(health_scores)
        
    def calc_loadings(self):
        
        #print table of indicator variable loadings
        load_table = []
        for i in range(self.n):
            load_table.append(self.var_load[:, i]) 

        features = self.extract_features()
        loadings = zip(features, *load_table)
        with open(self.loadings_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(loadings)

def main():
    health_obj = HealthScores(cols_filepath="data/all_data_std.csv", data_filepath="data/all_data_std.csv",\
        pca_filepath = "output/pca_all_std.csv", loadings_filepath="output/loadings_all_std.csv")
    health_obj.calc_pca()
    health_obj.calc_loadings()

if __name__ == '__main__':
    main()