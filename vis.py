import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def pca_std():
	data = pd.read_csv('output/pca_1_std.csv',names=['Town','Health_Score'])
	data.set_index('Town',inplace=True)
	t_3 = data.iloc[:3]
	b_3 = data.iloc[-3:]
	v_1 = pd.concat([t_3,b_3])
	plt.bar(x=v_1.index,height=v_1['Health_Score'])
	plt.show()

def pca_mn():
	data = pd.read_csv('output/pca_1_mn.csv',names=['Town','Health_Score'])
	data.set_index('Town',inplace=True)
	t_3 = data.iloc[:3]
	b_3 = data.iloc[-3:]
	v_1 = pd.concat([t_3,b_3])
	plt.bar(x=v_1.index,height=v_1['Health_Score'])
	plt.show()

pca_std()
pca_mn()