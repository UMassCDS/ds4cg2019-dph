import pandas as pd
import numpy as np 

def extract_features(data):
	"""
	To extract the column headers so we can map PC components to relevant factors
	"""
	columns = [x for x in enumerate(data.columns, 1)]
	print(columns)

def extract_towns(data):
	"""
	Extract the town names, so that health score can have meaning
	"""
	towns = [x for x in enumerate(list(data.index))]
	print(towns)
	
data = pd.read_csv("Determinants (std).csv",index_col=0)
extract_features(data)
extract_towns(data)