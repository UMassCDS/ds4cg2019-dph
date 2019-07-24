import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

def visualize_pca(file_std, file_mn, file_all_std, file_all_mn, n):
	barWidth = 0.3
	fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(15,7))

	d0 = pd.read_csv(file_std,names=['Town','Health_Score'])
	d0.set_index('Town',inplace=True)
	t0 = d0.iloc[:n]
	b0 = d0.iloc[-n:]
	v0 = pd.concat([t0,b0])
	ax0.bar(x=v0.index,height=v0['Health_Score'],width=barWidth, align='center', color='r')
	ax0.set_title('Standardized Data - Determinants only')

	d1 = pd.read_csv(file_mn,names=['Town','Health_Score'])
	d1.set_index('Town',inplace=True)
	t1 = d1.iloc[:n]
	b1= d1.iloc[-n:]
	v1 = pd.concat([t1,b1])
	ax1.bar(x=v1.index,height=v1['Health_Score'],width=barWidth, align='center',color='r')
	ax1.set_title('Mean Normalized Data - Determinants only')
	
	d2 = pd.read_csv(file_all_std,names=['Town','Health_Score'])
	d2.set_index('Town',inplace=True)
	t2 = d2.iloc[:n]
	b2= d2.iloc[-n:]
	v2 = pd.concat([t2,b2])
	ax2.bar(x=v2.index,height=v2['Health_Score'],width=barWidth, align='center',color='r')
	ax2.set_title('Standardized Data')

	d3 = pd.read_csv(file_all_mn,names=['Town','Health_Score'])
	d3.set_index('Town',inplace=True)
	t3 = d3.iloc[:n]
	b3= d3.iloc[-n:]
	v3 = pd.concat([t3,b3])
	ax3.bar(x=v3.index,height=v3['Health_Score'],width=barWidth, align='center',color='r')
	ax3.set_title('Mean Normalized Data')

	plt.tight_layout()
	plt.show()

def visualize_domains(file_path, n, ver):
	domain_data = pd.read_csv(file_path, index_col=0).sort_values(by='AVG')
	domain_names = list(domain_data)

	bot_domains = domain_data.head(n)
	bot_names = list(bot_domains.index)
	bot_data = np.array(bot_domains).T

	barWidth = 0.075

	r1 = np.arange(n)
	r2 = [x+barWidth for x in r1]
	r3 = [x+barWidth for x in r2]
	r4 = [x+barWidth for x in r3]
	r5 = [x+barWidth for x in r4]
	r6 = [x+barWidth for x in r5]
	r7 = [x+barWidth for x in r6]
	r8 = [x+barWidth for x in r7]
	r9 = [x+barWidth for x in r8]

	r = [r1, r2, r3, r4, r5, r6, r7, r8, r9]

	plt.figure(figsize=(12,6))

	for t,r_i in enumerate(r):
		plt.bar(r_i, bot_data[t,:], width=barWidth, label = domain_names[t], align='center')
	
	plt.xticks([num + barWidth for num in range(n)], bot_names)
	plt.xlabel('Municipality')
	plt.ylabel('Health Score')
	plt.ylim(top=1)
	plt.title('Health Scores per Domain for the Lowest Scored Towns with ' + ver + ' Data')
	plt.legend()
	plt.savefig('visualizations/score_per_domain_' + ver.lower())

def main():
	# visualize_domains("output/pca_domains_std.csv", 5, 'STD')
	# visualize_domains("output/pca_domains_mn.csv", 5, 'MN')
	visualize_pca('output/pca_1_std.csv','output/pca_1_mn.csv','output/pca_all_std.csv','output/pca_all_mn.csv',3)

if __name__ == "__main__":
	main() 
