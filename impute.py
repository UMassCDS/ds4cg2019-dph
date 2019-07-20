import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

# data = pd.read_csv('../Determinants (std).csv')
data = pd.read_csv('../Determinants_mn.csv')

data.set_index('index', inplace = True)

output = IterativeImputer(estimator = BayesianRidge(), sample_posterior=True).fit_transform(data)

df = pd.DataFrame(output)

df.to_csv('determinant_data_mn.csv')

data = pd.read_csv('../health_outcomes_mn.csv')
data.set_index('index', inplace = True)

output = IterativeImputer(estimator = BayesianRidge(), sample_posterior=True).fit_transform(data)

df = pd.DataFrame(output)

df.to_csv('outcome_data_mn.csv')