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
plt.rcParams['figure.figsize'] = (11, 7)

def visualize_scores(locations=['Boston', 'Amherst', 'Springfield']):
    
    """ Creates bar plots for every town in ``locations`` """
    
    # read data
    ALL_MC = pd.read_csv(os.getcwd()+'/output/pca_all_mc.csv', names=['Town', 'ALL_MC'], skiprows=1)
    ALL_MC['rank_ALL_MC'] = ALL_MC.index+1
    ALL_MC.set_index('Town', inplace=True)
    
    ALL_STD = pd.read_csv(os.getcwd()+'/output/pca_all_std.csv', names=['Town', 'ALL_STD'], skiprows=1)
    ALL_STD['rank_ALL_STD'] = ALL_STD.index+1
    ALL_STD.set_index('Town', inplace=True)
    
    DET_MC = pd.read_csv(os.getcwd()+'/output/pca_determinant_mc.csv', names=['Town', 'DET_MC'], skiprows=1)
    DET_MC['rank_DET_MC'] = DET_MC.index+1
    DET_MC.set_index('Town', inplace=True)
    
    DET_STD = pd.read_csv(os.getcwd()+'/output/pca_determinant_std.csv', names=['Town', 'DET_STD'], skiprows=1)
    DET_STD['rank_DET_STD'] = DET_STD.index+1
    DET_STD.set_index('Town', inplace=True)
    
    OUT_MC = pd.read_csv(os.getcwd()+'/output/pca_outcome_mc.csv', names=['Town', 'OUT_MC'], skiprows=1)
    OUT_MC['rank_OUT_MC'] = OUT_MC.index+1
    OUT_MC.set_index('Town', inplace=True)
    
    OUT_STD = pd.read_csv(os.getcwd()+'/output/pca_outcome_std.csv', names=['Town', 'OUT_STD'], skiprows=1)
    OUT_STD['rank_OUT_STD'] = OUT_STD.index+1
    OUT_STD.set_index('Town', inplace=True)
     
    ALL_MC_DEC = pd.read_csv(os.getcwd()+'//output/pca_decorrelated_all_mc.csv', names=['Town', 'ALL_MC_DEC'], skiprows=1)
    ALL_MC_DEC['rank_ALL_MC_DEC'] = ALL_MC_DEC.index+1
    ALL_MC_DEC.set_index('Town', inplace=True)
    
    ALL_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_all_std.csv', names=['Town', 'ALL_STD_DEC'], skiprows=1)
    ALL_STD_DEC['rank_ALL_STD_DEC'] = ALL_STD_DEC.index+1
    ALL_STD_DEC.set_index('Town', inplace=True)
    
    DET_MC_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorreleated_determinant_mc.csv', names=['Town', 'DET_MC_DEC'], skiprows=1)
    DET_MC_DEC['rank_DET_MC_DEC'] = DET_MC_DEC.index+1
    DET_MC_DEC.set_index('Town', inplace=True)
    
    DET_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_determinant_std.csv', names=['Town', 'DET_STD_DEC'], skiprows=1)
    DET_STD_DEC['rank_DET_STD_DEC'] = DET_STD_DEC.index+1
    DET_STD_DEC.set_index('Town', inplace=True)
    
    OUT_MC_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_outcome_mc.csv', names=['Town', 'OUT_MC_DEC'], skiprows=1)
    OUT_MC_DEC['rank_OUT_MC_DEC'] = OUT_MC_DEC.index+1
    OUT_MC_DEC.set_index('Town', inplace=True)
    
    OUT_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_outcome_std.csv', names=['Town', 'OUT_STD_DEC'], skiprows=1)
    OUT_STD_DEC['rank_OUT_STD_DEC'] = OUT_STD_DEC.index+1
    OUT_STD_DEC.set_index('Town', inplace=True)
     
    DOM_MC = pd.read_csv(os.getcwd()+'/output/pca_domains_mc.csv',\
                          names=['index', 'BE_MC', 'CC_MC', 'ECON_MC', 'EDU_MC', 'EMP_MC', 'HEA_MC', 'HOU_MC', 'VIO_MC', 'AVG_MC'],\
                          index_col='index', skiprows=1)
    DOM_STD = pd.read_csv(os.getcwd()+'/output/pca_domains_std.csv',\
                           names=['index', 'BE_STD', 'CC_STD', 'ECON_STD', 'EDU_STD', 'EMP_STD', 'HEA_STD', 'HOU_STD', 'VIO_STD', 'AVG_STD'],\
                           index_col='index', skiprows=1)
    DOM_MC_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_domains_mc.csv', \
                            names=['index', 'BE_MC_DEC', 'CC_MC_DEC', 'ECON_MC_DEC', 'EDU_MC_DEC', 'EMP_MC_DEC', 'HEA_MC_DEC', 'HOU_MC_DEC', 'VIO_MC_DEC', 'AVG_MC_DEC'],\
                            index_col='index', skiprows=1)
    DOM_STD_DEC = pd.read_csv(os.getcwd()+'/output/pca_decorrelated_domains_std.csv', \
                            names=['index', 'BE_STD_DEC', 'CC_STD_DEC', 'ECON_STD_DEC', 'EDU_STD_DEC', 'EMP_STD_DEC', 'HEA_STD_DEC', 'HOU_STD_DEC', 'VIO_STD_DEC', 'AVG_STD_DEC'],\
                            index_col='index', skiprows=1)
    
    # create subsets 
    ALL = ALL_MC.join(ALL_STD, how='inner').join(ALL_MC_DEC, how='inner').join(ALL_STD_DEC, how='inner')
    DET = DET_MC.join(DET_STD, how='inner').join(DET_MC_DEC, how='inner').join(DET_STD_DEC, how='inner')
    OUT = OUT_MC.join(OUT_STD, how='inner').join(OUT_MC_DEC, how='inner').join(OUT_STD_DEC, how='inner')
    
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
        dom_df = pd.DataFrame({'Mean Normalized':{'Built Environment':DOM_MC.loc[location].values[0],
                                                   'Community Context':DOM_MC.loc[location].values[1],
                                                   'Economy':DOM_MC.loc[location].values[2],
                                                   'Education':DOM_MC.loc[location].values[3],
                                                   'Employment':DOM_MC.loc[location].values[4],
                                                   'Health':DOM_MC.loc[location].values[5],
                                                   'Housing':DOM_MC.loc[location].values[6],
                                                   'Violence':DOM_MC.loc[location].values[7]},
                               'Standardized':{'Built Environment':DOM_STD.loc[location].values[0],
                                               'Community Context':DOM_STD.loc[location].values[1],
                                               'Economy':DOM_STD.loc[location].values[2],
                                               'Education':DOM_STD.loc[location].values[3],
                                               'Employment':DOM_STD.loc[location].values[4],
                                               'Health':DOM_STD.loc[location].values[5],
                                               'Housing':DOM_STD.loc[location].values[6],
                                               'Violence':DOM_STD.loc[location].values[7]},
                               'Mean Normalized(Decorrelated)':{'Built Environment':DOM_MC_DEC.loc[location].values[0],
                                                                'Community Context':DOM_MC_DEC.loc[location].values[1],
                                                                'Economy':DOM_MC_DEC.loc[location].values[2],
                                                                'Education':DOM_MC_DEC.loc[location].values[3],
                                                                'Employment':DOM_MC_DEC.loc[location].values[4],
                                                                'Health':DOM_MC_DEC.loc[location].values[5],
                                                                'Housing':DOM_MC_DEC.loc[location].values[6],
                                                                'Violence':DOM_MC_DEC.loc[location].values[7]},
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