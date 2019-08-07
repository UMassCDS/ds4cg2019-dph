#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:29:47 2019

@author: roshanprakash
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 7)

def visualize_scores(locations=['Boston', 'Amherst', 'Springfield']):
    
    """ Creates bar plots for every town in ``locations`` """
    
    # read data
    ALL_MN = pd.read_csv(os.getcwd()+'/output/pca_all_mn.csv', names=['Town', 'ALL_MN'], skiprows=1)
    ALL_MN['rank_ALL_MN'] = ALL_MN.index+1
    ALL_MN.set_index('Town', inplace=True)
    
    ALL_STD = pd.read_csv(os.getcwd()+'/output/pca_all_std.csv', names=['Town', 'ALL_STD'], skiprows=1)
    ALL_STD['rank_ALL_STD'] = ALL_STD.index+1
    ALL_STD.set_index('Town', inplace=True)
    
    DET_MN = pd.read_csv(os.getcwd()+'/output/pca_determinant_mn.csv', names=['Town', 'DET_MN'], skiprows=1)
    DET_MN['rank_DET_MN'] = DET_MN.index+1
    DET_MN.set_index('Town', inplace=True)
    
    DET_STD = pd.read_csv(os.getcwd()+'/output/pca_determinant_std.csv', names=['Town', 'DET_STD'], skiprows=1)
    DET_STD['rank_DET_STD'] = DET_STD.index+1
    DET_STD.set_index('Town', inplace=True)
    
    OUT_MN = pd.read_csv(os.getcwd()+'/output/pca_outcome_mn.csv', names=['Town', 'OUT_MN'], skiprows=1)
    OUT_MN['rank_OUT_MN'] = OUT_MN.index+1
    OUT_MN.set_index('Town', inplace=True)
    
    OUT_STD = pd.read_csv(os.getcwd()+'/output/pca_outcome_std.csv', names=['Town', 'OUT_STD'], skiprows=1)
    OUT_STD['rank_OUT_STD'] = OUT_STD.index+1
    OUT_STD.set_index('Town', inplace=True)
     
    ALL_MN_DEC = pd.read_csv(os.getcwd()+'//output/pca_decorrelated_all_mn.csv', names=['Town', 'ALL_MN_DEC'], skiprows=1)
    ALL_MN_DEC['rank_ALL_MN_DEC'] = ALL_MN_DEC.index+1
    ALL_MN_DEC.set_index('Town', inplace=True)
    
    ALL_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_all_std.csv', names=['Town', 'ALL_STD_DEC'], skiprows=1)
    ALL_STD_DEC['rank_ALL_STD_DEC'] = ALL_STD_DEC.index+1
    ALL_STD_DEC.set_index('Town', inplace=True)
    
    DET_MN_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorreleated_determinant_mn.csv', names=['Town', 'DET_MN_DEC'], skiprows=1)
    DET_MN_DEC['rank_DET_MN_DEC'] = DET_MN_DEC.index+1
    DET_MN_DEC.set_index('Town', inplace=True)
    
    DET_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_determinant_std.csv', names=['Town', 'DET_STD_DEC'], skiprows=1)
    DET_STD_DEC['rank_DET_STD_DEC'] = DET_STD_DEC.index+1
    DET_STD_DEC.set_index('Town', inplace=True)
    
    OUT_MN_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_outcome_mn.csv', names=['Town', 'OUT_MN_DEC'], skiprows=1)
    OUT_MN_DEC['rank_OUT_MN_DEC'] = OUT_MN_DEC.index+1
    OUT_MN_DEC.set_index('Town', inplace=True)
    
    OUT_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_outcome_std.csv', names=['Town', 'OUT_STD_DEC'], skiprows=1)
    OUT_STD_DEC['rank_OUT_STD_DEC'] = OUT_STD_DEC.index+1
    OUT_STD_DEC.set_index('Town', inplace=True)
     
    DOM_MN = pd.read_csv(os.getcwd()+'/output/pca_domains_mn.csv',\
                          names=['index', 'BE_MN', 'CC_MN', 'ECON_MN', 'EDU_MN', 'EMP_MN', 'HEA_MN', 'HOU_MN', 'VIO_MN', 'AVG_MN'],\
                          index_col='index', skiprows=1)
    DOM_STD = pd.read_csv(os.getcwd()+'/output/pca_domains_std.csv',\
                           names=['index', 'BE_STD', 'CC_STD', 'ECON_STD', 'EDU_STD', 'EMP_STD', 'HEA_STD', 'HOU_STD', 'VIO_STD', 'AVG_STD'],\
                           index_col='index', skiprows=1)
    DOM_MN_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_domains_mn.csv', \
                            names=['index', 'BE_MN_DEC', 'CC_MN_DEC', 'ECON_MN_DEC', 'EDU_MN_DEC', 'EMP_MN_DEC', 'HEA_MN_DEC', 'HOU_MN_DEC', 'VIO_MN_DEC', 'AVG_MN_DEC'],\
                            index_col='index', skiprows=1)
    DOM_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_domains_std.csv', \
                            names=['index', 'BE_STD_DEC', 'CC_STD_DEC', 'ECON_STD_DEC', 'EDU_STD_DEC', 'EMP_STD_DEC', 'HEA_STD_DEC', 'HOU_STD_DEC', 'VIO_STD_DEC', 'AVG_STD_DEC'],\
                            index_col='index', skiprows=1)
    
    # create subsets 
    ALL = ALL_MN.join(ALL_STD, how='inner').join(ALL_MN_DEC, how='inner').join(ALL_STD_DEC, how='inner')
    DET = DET_MN.join(DET_STD, how='inner').join(DET_MN_DEC, how='inner').join(DET_STD_DEC, how='inner')
    OUT = OUT_MN.join(OUT_STD, how='inner').join(OUT_MN_DEC, how='inner').join(OUT_STD_DEC, how='inner')
    
    for location in locations:
        
        vals_ALL = ALL.loc[location]
        vals_DET = DET.loc[location]
        vals_OUT = OUT.loc[location]
        
        # plots across subsets
        sub_df = pd.DataFrame({'Mean Normalized':{'Determinants data':vals_DET.values[0], \
                                                   'Outcomes data':vals_OUT.values[0],
                                                   'All data':vals_ALL.values[0]}, \
                               'Standardized':{'Determinants data':vals_DET.values[2], \
                                               'Outcomes data':vals_OUT.values[2],
                                               'All data':vals_ALL.values[2]},
                               'Mean Normalized(Decorrelated)':{'Determinants data':vals_DET.values[4], \
                                                                'Outcomes data':vals_OUT.values[4],
                                                                'All data':vals_ALL.values[4]},
                               'Standardized(Decorrelated)':{'Determinants data':vals_DET.values[6], \
                                                             'Outcomes data':vals_OUT.values[6],
                                                             'All data':vals_ALL.values[6]}})
        # colors can be changed below
        ax = sub_df.plot(kind='bar', width=0.4, edgecolor='black', alpha=0.9, colors=['green', 'red', 'black', 'pink'])
        plt.title('Health scores for {}, across different subsets of data'.format(location))
        plt.xlabel('Data subset')
        plt.ylabel('Health Score')
        plt.xticks(rotation=0)
        y = [vals_ALL.values[0], vals_ALL.values[2], vals_ALL.values[4], vals_ALL.values[6], \
             vals_DET.values[0], vals_DET.values[2], vals_DET.values[4], vals_DET.values[6], \
             vals_OUT.values[0], vals_OUT.values[2], vals_OUT.values[4], vals_OUT.values[6]]
        ranks = [vals_ALL.values[1], vals_ALL.values[3], vals_ALL.values[5], vals_ALL.values[7], \
                 vals_DET.values[1], vals_DET.values[3], vals_DET.values[5], vals_DET.values[7], \
                 vals_OUT.values[1], vals_OUT.values[3], vals_OUT.values[5], vals_OUT.values[7]]
        
        # DO NOT MODIFY THIS!
        offset=0.0
        for i, v in enumerate(y):
            ax.text((i*0.1)-0.2+offset, v+0.01, int(ranks[i]), color='black', fontweight='bold', fontsize=9)
            if (i+1)%4==0:
                offset+=0.605
        
        # plots across domains
        dom_df = pd.DataFrame({'Mean Normalized':{'Built Environment':DOM_MN.loc[location].values[0],
                                                   'Community Context':DOM_MN.loc[location].values[1],
                                                   'Economy':DOM_MN.loc[location].values[2],
                                                   'Education':DOM_MN.loc[location].values[3],
                                                   'Employment':DOM_MN.loc[location].values[4],
                                                   'Health':DOM_MN.loc[location].values[5],
                                                   'Housing':DOM_MN.loc[location].values[6],
                                                   'Violence':DOM_MN.loc[location].values[7]},
                               'Standardized':{'Built Environment':DOM_STD.loc[location].values[0],
                                               'Community Context':DOM_STD.loc[location].values[1],
                                               'Economy':DOM_STD.loc[location].values[2],
                                               'Education':DOM_STD.loc[location].values[3],
                                               'Employment':DOM_STD.loc[location].values[4],
                                               'Health':DOM_STD.loc[location].values[5],
                                               'Housing':DOM_STD.loc[location].values[6],
                                               'Violence':DOM_STD.loc[location].values[7]},
                               'Mean Normalized(Decorrelated)':{'Built Environment':DOM_MN_DEC.loc[location].values[0],
                                                                'Community Context':DOM_MN_DEC.loc[location].values[1],
                                                                'Economy':DOM_MN_DEC.loc[location].values[2],
                                                                'Education':DOM_MN_DEC.loc[location].values[3],
                                                                'Employment':DOM_MN_DEC.loc[location].values[4],
                                                                'Health':DOM_MN_DEC.loc[location].values[5],
                                                                'Housing':DOM_MN_DEC.loc[location].values[6],
                                                                'Violence':DOM_MN_DEC.loc[location].values[7]},
                               'Standardized(Decorrelated)':{'Built Environment':DOM_STD_DEC.loc[location].values[0],
                                                             'Community Context':DOM_STD_DEC.loc[location].values[1],
                                                             'Economy':DOM_STD_DEC.loc[location].values[2],
                                                             'Education':DOM_STD_DEC.loc[location].values[3],
                                                             'Employment':DOM_STD_DEC.loc[location].values[4],
                                                             'Health':DOM_STD_DEC.loc[location].values[5],
                                                             'Housing':DOM_STD_DEC.loc[location].values[6],
                                                             'Violence':DOM_STD_DEC.loc[location].values[7]}})
        # colors can be changed below
        dom_df.plot(kind='bar', width=0.7, edgecolor='black', alpha=0.9, colors=['green', 'red', 'black', 'pink'])
        plt.title('Health scores for {}, across different domains'.format(location))
        plt.xlabel('Domain')
        plt.ylabel('Health Score')
        plt.xticks(rotation=15)
        plt.show()
        
if __name__=='__main__':
    visualize_scores()