import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_pca_determinants(file_std, file_mc, file_all_std, file_all_mc, n, visualize_file):
    barWidth = 0.3
    _, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2,2, figsize=(15,7))

    d0 = pd.read_csv(file_std,index_col=0)
    t0 = d0.iloc[:n]
    b0 = d0.iloc[-n:]
    v0 = pd.concat([t0,b0])
    ax0.bar(x=v0.index,height=v0['Health Score'],width=barWidth, align='center', color='r')
    ax0.set_title('Standardized Data - Determinants only')

    d1 = pd.read_csv(file_mc,index_col=0)
    t1 = d1.iloc[:n]
    b1= d1.iloc[-n:]
    v1 = pd.concat([t1,b1])
    ax1.bar(x=v1.index,height=v1['Health Score'],width=barWidth, align='center',color='r')
    ax1.set_title('Mean Normalized Data - Determinants only')
    
    d2 = pd.read_csv(file_all_std,index_col=0)
    t2 = d2.iloc[:n]
    b2= d2.iloc[-n:]
    v2 = pd.concat([t2,b2])
    ax2.bar(x=v2.index,height=v2['Health Score'],width=barWidth, align='center',color='r')
    ax2.set_title('Standardized Data')

    d3 = pd.read_csv(file_all_mc,index_col=0)
    t3 = d3.iloc[:n]
    b3= d3.iloc[-n:]
    v3 = pd.concat([t3,b3])
    ax3.bar(x=v3.index,height=v3['Health Score'],width=barWidth, align='center',color='r')
    ax3.set_title('Mean Normalized Data')

    plt.tight_layout()
    plt.savefig(visualize_file)

def visualize_pca_outcomes(file_std, file_mc, n, visualize_file):
    barWidth = 0.3
    _, ((ax0, ax1)) = plt.subplots(1,2, figsize=(15,7))

    d0 = pd.read_csv(file_std,index_col=0)
    t0 = d0.iloc[:n]
    b0 = d0.iloc[-n:]
    v0 = pd.concat([t0,b0])
    ax0.bar(x=v0.index,height=v0['Health Score'],width=barWidth, align='center', color='b')
    ax0.set_title('Standardized Data - Outcomes only')

    d1 = pd.read_csv(file_mc,index_col=0)
    t1 = d1.iloc[:n]
    b1= d1.iloc[-n:]
    v1 = pd.concat([t1,b1])
    ax1.bar(x=v1.index,height=v1['Health Score'],width=barWidth, align='center',color='b')
    ax1.set_title('Mean Normalized Data - Outcomes only')

    plt.tight_layout()
    plt.savefig(visualize_file)

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

def visualize_corr_mat(file_path, write_path, grid_size = 35, size_scale = 80, legend_fs = 24, label_fs = 24, label_rotation = 45):
    corr_mat = pd.read_csv(file_path, index_col=0)
    features = list(corr_mat)
    corr_mat = np.array(corr_mat)

    n = len(features)
    x = []
    y = []

    for _ in range(n):
        x = x + features
    
    for f in range(n):
        for _ in range(n):
            y = y + [features[f]]

    fig = plt.figure()
    fig.set_size_inches(grid_size, grid_size)
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)
    ax = plt.subplot(plot_grid[:,:-1])

    feat_to_num = {p[1]:p[0] for p in enumerate(features)}

    value = corr_mat.flatten()
    
    x = pd.Series(x)
    y = pd.Series(y)
    size = pd.Series(value).abs()
    value = pd.Series(value)

    ax.scatter(
        x = x.map(feat_to_num),
        y = y.map(feat_to_num),
        s = size * size_scale,
        c = value.apply(value_to_color),
        marker = 's'
    )

    ax.set_xticks([feat_to_num[v] for v in features])
    ax.set_xticklabels(features, rotation=label_rotation, horizontalalignment='right')
    ax.set_yticks([feat_to_num[v] for v in features])
    ax.set_yticklabels(features, rotation = label_rotation)
    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in feat_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in feat_to_num.values()]) + 0.5])
    ax.tick_params(labelsize=label_fs)

    ax = plt.subplot(plot_grid[:,-1])

    n_colors = 256
    palette = sns.diverging_palette(20, 220, n=n_colors)
    color_min, color_max = [-1, 1]

    col_x = [0]*len(palette)
    bar_y=np.linspace(color_min, color_max, n_colors)

    bar_height = bar_y[1] - bar_y[0]

    ax.barh(
        y=bar_y,
        width=[5]*len(palette),
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0,
        edgecolor = 'white'
    )
    
    ax.set_xlim(1, 2)
    ax.grid(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 9))
    ax.tick_params(labelsize=legend_fs)
    ax.yaxis.tick_right()

    plt.savefig(write_path)
    
def main():
    visualize_corr_mat('output/correlation_matrix_determinant.csv', 'visualizations/correlation_matrix_determinant_vis', label_fs = 10, legend_fs=36)
    visualize_corr_mat('output/correlation_matrix_all.csv', 'visualizations/correlation_matrix_all_vis',  grid_size = 40, size_scale = 50, label_fs=10, legend_fs=40)
    visualize_corr_mat('output/correlation_matrix_outcome.csv', 'visualizations/correlation_matrix_outcome_vis',  label_rotation=35, grid_size = 38, size_scale = 1300, legend_fs = 32)
    
    visualize_domains("output/pca_domains_std.csv", 5, 'STD')
    visualize_domains("output/pca_domains_mc.csv", 5, 'MC')
        
    compare_results('output/pca_determinant_mc.csv', 'output/pca_determinant_std.csv', 'visualizations/result_comparison_determinant', 'Determinant')
    compare_results('output/pca_all_mc.csv', 'output/pca_all_std.csv', 'visualizations/result_comparison_all', 'All')
    compare_results('output/pca_outcome_mc.csv', 'output/pca_outcome_std.csv', 'visualizations/result_comparison_outcome', 'Outcome')
    compare_results_per_domain('output/pca_domains_mc.csv', 'output/pca_domains_std.csv', 'visualizations/result_comparison_domain')

    visualize_pca_determinants('output/pca_determinant_std.csv','output/pca_determinant_mc.csv','output/pca_all_std.csv','output/pca_all_mc.csv',3,'visualizations/pca_high_low_3.png')
    visualize_pca_outcomes('output/pca_outcome_std.csv','output/pca_outcome_mc.csv',3,'visualizations/outcome_high_low_3.png')

    visualize_corr_mat('output/correlation_matrix_built_environment.csv', 'visualizations/correlation_matrix_built_environment_vis', label_fs = 26, label_rotation = 30, grid_size = 32, size_scale = 4000, legend_fs=32)
    visualize_corr_mat('output/correlation_matrix_community_context.csv', 'visualizations/correlation_matrix_community_context_vis', grid_size = 24, size_scale = 2000)
    visualize_corr_mat('output/correlation_matrix_economy.csv', 'visualizations/correlation_matrix_economy_vis', label_fs = 26, label_rotation = 40, grid_size = 38, size_scale = 7000, legend_fs= 34)
    visualize_corr_mat('output/correlation_matrix_education.csv', 'visualizations/correlation_matrix_education_vis', grid_size = 30, size_scale = 2000, legend_fs=30)
    visualize_corr_mat('output/correlation_matrix_employment.csv', 'visualizations/correlation_matrix_employment_vis', label_rotation = 40, grid_size = 26, size_scale = 900)
    visualize_corr_mat('output/correlation_matrix_healthcare_access.csv', 'visualizations/correlation_matrix_healthcare_access_vis', label_rotation = 30, grid_size = 40, size_scale = 8000, legend_fs=36)
    visualize_corr_mat('output/correlation_matrix_housing.csv', 'visualizations/correlation_matrix_housing_vis', grid_size = 45, size_scale = 2000, legend_fs =40)
    visualize_corr_mat('output/correlation_matrix_violence.csv', 'visualizations/correlation_matrix_violence_vis', grid_size = 32, size_scale = 30000, legend_fs=32)

    visualize_corr_mat('output/correlation_matrix_decorrelated_determinant.csv', 'visualizations/correlation_matrix_decorrelated_determinant_vis', label_fs = 12, legend_fs=36)
    visualize_corr_mat('output/correlation_matrix_decorrelated_all.csv', 'visualizations/correlation_matrix_decorrelated_all_vis', grid_size = 40, size_scale = 50, label_fs=12, legend_fs=40)
    visualize_corr_mat('output/correlation_matrix_decorrelated_outcome.csv', 'visualizations/correlation_matrix_decorrelated_outcome_vis', label_rotation=35, grid_size = 38, size_scale = 2000, legend_fs = 32)

    visualize_corr_mat('output/correlation_matrix_decorrelated_built_environment.csv', 'visualizations/correlation_matrix_decorrelated_built_environment_vis', label_fs = 26, label_rotation = 30, grid_size = 32, size_scale = 5000, legend_fs=32)
    visualize_corr_mat('output/correlation_matrix_decorrelated_community_context.csv', 'visualizations/correlation_matrix_decorrelated_community_context_vis', grid_size = 24, size_scale = 2000)
    visualize_corr_mat('output/correlation_matrix_decorrelated_economy.csv', 'visualizations/correlation_matrix_decorrelated_economy_vis', label_fs = 26, label_rotation = 40, grid_size = 38, size_scale = 17000, legend_fs= 34)
    visualize_corr_mat('output/correlation_matrix_decorrelated_education.csv', 'visualizations/correlation_matrix_decorrelated_education_vis', grid_size = 30, size_scale = 2000, legend_fs=30)
    visualize_corr_mat('output/correlation_matrix_decorrelated_employment.csv', 'visualizations/correlation_matrix_decorrelated_employment_vis', label_rotation = 40, grid_size = 26, size_scale = 1000)
    visualize_corr_mat('output/correlation_matrix_decorrelated_healthcare_access.csv', 'visualizations/correlation_matrix_decorrelated_healthcare_access_vis', label_rotation = 30, grid_size = 40, size_scale = 13000, legend_fs= 36)
    visualize_corr_mat('output/correlation_matrix_decorrelated_housing.csv', 'visualizations/correlation_matrix_decorrelated_housing_vis', grid_size = 45, size_scale = 3000, legend_fs = 40)
    visualize_corr_mat('output/correlation_matrix_decorrelated_violence.csv', 'visualizations/correlation_matrix_decorrelated_violence_vis', grid_size = 32, size_scale = 30000, legend_fs=32)

    compare_results('output/pca_decorreleated_determinant_mc.csv', 'output/pca_decorrelated_determinant_std.csv', 'visualizations/result_comparison_decorrelated_determinant', 'Decorrelated Determinant')
    compare_results('output/pca_decorrelated_all_mc.csv', 'output/pca_decorrelated_all_std.csv', 'visualizations/result_comparison_decorrelated_all', 'Decorrelated All')
    compare_results('output/pca_decorrelated_outcome_mc.csv', 'output/pca_decorrelated_outcome_std.csv', 'visualizations/result_comparison_decorrelated_outcome', 'Decorrelated Outcome')
    compare_results_per_domain('output/pca_decorrelated_domains_mc.csv', 'output/pca_decorrelated_domains_std.csv', 'visualizations/result_comparison_decorrelated_domain')
    
def value_to_color(val):
    n_colors = 256
    color_min, color_max = [-1, 1]
    palette = sns.diverging_palette(20, 220, n=n_colors)
    try:
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        res = palette[ind]
    except IndexError:
        if ind > 255:
            res = palette[255]
    return res

def compare_results(cov_file, corr_file, write_file, sub_set):
    cov = pd.read_csv(cov_file, index_col=0)
    cov.rename(columns={'Health Score':'COV'}, inplace=True)
    corr = pd.read_csv(corr_file, index_col=0)
    corr.rename(columns={'Health Score':'CORR'}, inplace=True)
    comp = cov.join(corr, on=['Town'], how = 'inner')

    fig = plt.figure(figsize=(10,10))
    
    plt.scatter(comp['COV'], comp['CORR'])
    plt.title('Comparison of Covariance and Correlation Matrix PCA Results for '+sub_set+' Subset')
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    plt.xlabel('Covariance Matrix Results')
    plt.ylabel('Correlation Matrix Results')
    plt.savefig(write_file)

def compare_results_per_domain(cov_file, corr_file, write_file):
    cov = pd.read_csv(cov_file, index_col=0)
    corr = pd.read_csv(corr_file, index_col=0)

    domains = np.array(list(cov)[:-1]).reshape((2,4))

    fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')
    fig.set_size_inches(20,10)
    fig.suptitle("Comparison of Covariance and Correlation Matrix PCA Results per Domain")

    for i in range(2):
        for j in range(4):
            dom = domains[i,j]
            ax[i,j].scatter(cov[dom], corr[dom])
            ax[i,j].set_title(dom)
    
    fig.text(0.5, 0.04, 'Covariance Matrix Results', ha='center', va='center')
    fig.text(0.06, 0.5, 'Correlation Matrix Results', ha='center', va='center', rotation='vertical')

    plt.savefig(write_file)

if __name__ == "__main__":
    main() 
