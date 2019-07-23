import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
'''
data = pd.read_csv('output/pca_1_std.csv',index_col=0)
print(data.columns)
# print(data.head())
# plt.bar(x = data.index, height=data[1])
'''

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
    visualize_domains("output/pca_domains_std.csv", 5, 'STD')
    visualize_domains("output/pca_domains_mn.csv", 5, 'MN')

if __name__ == "__main__":
    main() 