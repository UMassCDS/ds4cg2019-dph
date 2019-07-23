import pandas as pd
import numpy as np
import csv

det = pd.read_csv('data/determinant_data_mn.csv',index_col=0)
det_columns = pd.read_csv('data/clean_determinants_mn.csv', index_col=0).columns
out = pd.read_csv('data/outcome_data_mn.csv', index_col=0)
out_columns = pd.read_csv('data/health_outcomes_mn.csv', index_col=0).columns

all_cols = list(det_columns) + list(out_columns)

full = pd.concat([det, out], axis=1, sort=False)
full.columns = all_cols
print(full.head())
full.to_csv('data/all_data_mn.csv')

