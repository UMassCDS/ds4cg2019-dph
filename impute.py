import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

'''
Methods to impute the data following sklearn library built-in imputation methods 
'''
# data = pd.read_csv('data/Determinants (std).csv')
data = pd.read_csv('data/determinant_negated.csv')

#data_drop = ['pop_bachelorsdegree_%', 'risk_per1000']

data.set_index('index', inplace = True)

#data = data.drop(data_drop, axis = 1)

#data.to_csv('data/clean_determinants.csv')

#To impute using Bayesian Ridge
output = IterativeImputer(estimator = BayesianRidge(), sample_posterior=True).fit_transform(data)

df = pd.DataFrame(output)

df.to_csv('data/health_determinant.csv')

'''
#To impute outcomes 
data = pd.read_csv('../health_outcomes.csv')
#data = pd.read_csv('../health_outcomes_mn.csv')
data.set_index('index', inplace = True)

output = IterativeImputer(estimator = BayesianRidge(), sample_posterior=True).fit_transform(data)

df = pd.DataFrame(output)

df.to_csv('outcome_data_mn.csv')
'''
