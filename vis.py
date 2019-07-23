import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

data = pd.read_csv('output/pca_1_std.csv',index_col=0)
print(data.columns)
# print(data.head())
# plt.bar(x = data.index, height=data[1])