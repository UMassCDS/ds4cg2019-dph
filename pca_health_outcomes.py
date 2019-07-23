import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA
from pca import HealthScores

def main():
    health_obj = HealthScores(cols_filepath="data/health_outcomes_std.csv", data_filepath="data/outcome_data_std.csv", pca_filepath = "output/pca_health_outcomes_std.csv")
    health_obj.calc_pca()

    health_obj = HealthScores(cols_filepath="data/health_outcomes_mn.csv", data_filepath="data/outcome_data_mn.csv", pca_filepath = "output/pca_health_outcomes_mn.csv")
    health_obj.calc_pca()

if __name__ == '__main__':
    main()

