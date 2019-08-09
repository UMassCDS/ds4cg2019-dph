from pca import HealthScores
import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

DETERMINANT_STD = {'cols_filepath':"data/health_determinants.csv", 
                    'data_filepath':"data/determinant_data_std.csv",
                    'pca_filepath':"output/pca_determinant_std.csv", 
                    'loadings_filepath':"output/loadings_determinant_std.csv",
                    'domain_filepath':"output/pca_domains_std.csv",
                    'corrmat_filepath':"output/correlation_matrix_determinant.csv", 
                    'pvalue_filepath':"output/p_values_determinant.csv", 
                    'fa_filepath':"output/fa_determinant_std.csv",
                    'sigcorr_filepath':"output/significant_correlations_determinant.csv",
                    'decorrelated_filepath':"data/decorrelated_determinant_data_std",
                    'VER':'std',
                    'DC':False}

DETERMINANT_MC = {'cols_filepath':"data/health_determinants.csv",
                    'data_filepath':"data/determinant_data_mc.csv",
                    'pca_filepath':"output/pca_determinant_mc.csv", 
                    'loadings_filepath':"output/loadings_determinant_mc.csv", 
                    'domain_filepath':"output/pca_domains_mc.csv",
                    'corrmat_filepath':"output/correlation_matrix_determinant.csv",
                    'pvalue_filepath':"output/p_values_determinant.csv",
                    'fa_filepath':"output/fa_determinant_mc.csv", 
                    'sigcorr_filepath':"output/significant_correlations_determinant.csv",
                    'decorrelated_filepath':'data/decorrelated_determinant_data_mc',
                    'VER':'mc',
                    'DC':False}

OUTCOME_STD = {'cols_filepath':"data/health_outcomes.csv",
                'data_filepath':"data/outcome_data_std.csv",
                'pca_filepath':"output/pca_outcome_std.csv", 
                'loadings_filepath':"output/loadings_outcome_std.csv",
                'domain_filepath':None,
                'corrmat_filepath':"output/correlation_matrix_outcome.csv", 
                'pvalue_filepath':"output/p_values_outcome.csv",
                'fa_filepath':"output/fa_outcome_std.csv", 
                'sigcorr_filepath':'output/significant_correlations_outcome.csv',
                'decorrelated_filepath':'data/decorrelated_outcome_data_std',
                'VER':'std',
                'DC':False}

OUTCOME_MC = {'cols_filepath':"data/health_outcomes.csv", 
                'data_filepath':"data/outcome_data_mc.csv",
                'pca_filepath':"output/pca_outcome_mc.csv",
                'loadings_filepath':"output/loadings_outcome_mc.csv",
                'domain_filepath':None,
                'corrmat_filepath':"output/correlation_matrix_outcome.csv", 
                'pvalue_filepath':"output/p_values_outcome.csv",
                'fa_filepath':"output/fa_outcome_mc.csv",
                'sigcorr_filepath':'output/significant_correlations_outcome.csv',
                'decorrelated_filepath':'data/decorrelated_outcome_data_mc',
                'VER':'mc',
                'DC':False}

ALL_STD = {'cols_filepath':"data/all_data.csv",
            'data_filepath':"data/all_data_std.csv",
            'pca_filepath':"output/pca_all_std.csv",
            'loadings_filepath':"output/loadings_all_std.csv",
            'domain_filepath':None,
            'corrmat_filepath':"output/correlation_matrix_all.csv",
            'pvalue_filepath':"output/p_values_all.csv", 
            'fa_filepath':"output/fa_all_std.csv",
            'sigcorr_filepath':"output/significant_correlations_all.csv",
            'decorrelated_filepath':"data/decorrelated_all_data_std",
            'VER':'std',
            'DC':False}

ALL_MC = {'cols_filepath':"data/all_data.csv",
            'data_filepath':"data/all_data_mc.csv",
            'pca_filepath':"output/pca_all_mc.csv",
            'loadings_filepath':"output/loadings_all_mc.csv",
            'domain_filepath':None,
            'corrmat_filepath':"output/correlation_matrix_all.csv",
            'pvalue_filepath':"output/p_values_all.csv",
            'fa_filepath':"output/fa_all_mc.csv",
            'sigcorr_filepath':"output/significant_correlations_all.csv", 
            'decorrelated_filepath':"data/decorrelated_all_data_mc",
            'VER':'mc',
            'DC':False}

DC_DETERMINANT_STD = {'cols_filepath':"data/decorrelated_determinant_data_std_columns.csv",
                        'data_filepath':"data/decorrelated_determinant_data_std.csv",
                        'pca_filepath':"output/pca_decorrelated_determinant_std.csv",
                        'loadings_filepath':"output/loadings_decorrelated_determinant_std.csv",
                        'domain_filepath':"output/pca_decorrelated_domains_std.csv",
                        'corrmat_filepath':"output/correlation_matrix_decorrelated_determinant.csv",
                        'pvalue_filepath':"output/p_values_decorrelated_determinant.csv",
                        'fa_filepath':'output/fa_decorrelated_determinant_std.csv',
                        'sigcorr_filepath':"output/significant_correlations_decorrelated_determinant.csv",
                        'decorrelated_filepath':None,
                        'VER':'std',
                        'DC':True}

DC_DETERMINANT_MC = {'cols_filepath':"data/decorrelated_determinant_data_mc_columns.csv",
                    'data_filepath':"data/decorrelated_determinant_data_mc.csv",
                    'pca_filepath':"output/pca_decorreleated_determinant_mc.csv", 
                    'loadings_filepath':"output/loadings_decorrelated_determinant_mc.csv", 
                    'domain_filepath':"output/pca_decorrelated_domains_mc.csv",
                    'corrmat_filepath':"output/correlation_matrix_decorrelaed_determinant.csv",
                    'pvalue_filepath':"output/p_values_decorrelated_determinant.csv",
                    'fa_filepath':'output/fa_decorrelated_determinant_mc.csv', 
                    'sigcorr_filepath':"output/significant_correlations_decorrelated_determinant.csv",
                    'decorrelated_filepath':None,
                    'VER':'mc',
                    'DC':True}

DC_OUTCOME_STD = {'cols_filepath':"data/decorrelated_outcome_data_std_columns.csv",
                'data_filepath':"data/decorrelated_outcome_data_std.csv",
                'pca_filepath':"output/pca_decorrelated_outcome_std.csv", 
                'loadings_filepath':"output/loadings_decorrelated_outcome_std.csv",
                'domain_filepath':None,
                'corrmat_filepath':"output/correlation_matrix_decorrelated_outcome.csv", 
                'pvalue_filepath':"output/p_values_decorrelated_outcome.csv",
                'fa_filepath':'output/fa_decorrelated_outcome_std.csv', 
                'sigcorr_filepath':'output/significant_correlations_decorrelated_outcome.csv',
                'decorrelated_filepath':None,
                'VER':'std',
                'DC':True}

DC_OUTCOME_MC = {'cols_filepath':"data/decorrelated_outcome_data_mc_columns.csv", 
                'data_filepath':"data/decorrelated_outcome_data_mc.csv",
                'pca_filepath':"output/pca_decorrelated_outcome_mc.csv",
                'loadings_filepath':"output/loadings_decorrelated_outcome_mc.csv",
                'domain_filepath':None,
                'corrmat_filepath':"output/correlation_matrix_decorrelated_outcome.csv", 
                'pvalue_filepath':"output/p_values_decorrelated_outcome.csv",
                'fa_filepath':'output/fa_decorrelated_outcome_mc.csv',
                'sigcorr_filepath':'output/significant_correlations_decorrelated_outcome.csv',
                'decorrelated_filepath':None,
                'VER':'mc',
                'DC':True}

DC_ALL_STD = {'cols_filepath':"data/decorrelated_all_data_std_columns.csv",
            'data_filepath':"data/decorrelated_all_data_std.csv",
            'pca_filepath':"output/pca_decorrelated_all_std.csv",
            'loadings_filepath':"output/loadings_decorrelated_all_std.csv",
            'domain_filepath':None,
            'corrmat_filepath':"output/correlation_matrix_decorrelated_all.csv",
            'pvalue_filepath':"output/p_values_decorrelated_all.csv", 
            'fa_filepath':'output/fa_decorrelated_all_std.csv',
            'sigcorr_filepath':"output/significant_correlations_decorrelated_all.csv",
            'decorrelated_filepath':None,
            'VER':'std',
            'DC':True}

DC_ALL_MC = {'cols_filepath':"data/decorrelated_all_data_mc_columns.csv",
            'data_filepath':"data/decorrelated_all_data_mc.csv",
            'pca_filepath':"output/pca_decorrelated_all_mc.csv",
            'loadings_filepath':"output/loadings_decorrealted_all_mc.csv",
            'domain_filepath':None,
            'corrmat_filepath':"output/correlation_matrix_decorrelated_all.csv",
            'pvalue_filepath':"output/p_values_decorrealted_all.csv",
            'fa_filepath':'output/fa_decorrelated_all_mc.csv',
            'sigcorr_filepath':"output/significant_correlations_decorrelated_all.csv", 
            'decorrelated_filepath': None,
            'VER':'mc',
            'DC':True}

def generate_results():
    health_obj = HealthScores(DETERMINANT_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings() 
    health_obj.factor_analysis(write=True)
    health_obj.score_per_domain()
    health_obj.factor_analysis_perdomain(write=True)
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_corr_mat_per_dom()
    health_obj.write_p_values_per_dom()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    for d in health_obj.domains:
        health_obj.write_significant_correlations('output/correlation_matrix_'+ d + '.csv', 'output/p_values_'+d+'.csv', 'output/significant_correlations_'+d+'.csv')
    health_obj.correlation_analysis()

    health_obj = HealthScores(DETERMINANT_MC)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.score_per_domain()
    health_obj.factor_analysis_perdomain(write=True)
    health_obj.correlation_analysis()
    
    health_obj = HealthScores(OUTCOME_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    health_obj.correlation_analysis()
    
    health_obj = HealthScores(OUTCOME_MC)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.correlation_analysis()

    health_obj = HealthScores(ALL_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    health_obj.correlation_analysis()

    health_obj = HealthScores(ALL_MC)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.correlation_analysis()

def generate_decorrelated_results():
    health_obj = HealthScores(DC_DETERMINANT_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.score_per_domain()
    health_obj.factor_analysis_perdomain(write=True)
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_corr_mat_per_dom()
    health_obj.write_p_values_per_dom()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    for d in health_obj.domains:
        health_obj.write_significant_correlations('output/correlation_matrix_decorrelated_'+ d + '.csv', 'output/p_values_decorrelated_'+d+'.csv', 'output/significant_correlations_decorrelated_'+d+'.csv')
    
    health_obj = HealthScores(DC_DETERMINANT_MC)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.score_per_domain()
    health_obj.factor_analysis_perdomain(write=True)
    
    health_obj = HealthScores(DC_OUTCOME_STD)
    health_obj.calc_pca(write=True)
    health_obj.factor_analysis(write=True)
    health_obj.calc_loadings()
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)

    health_obj = HealthScores(DC_OUTCOME_MC)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)

    health_obj = HealthScores(DC_ALL_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)

    health_obj = HealthScores(DC_ALL_MC)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.factor_analysis(write=True)

def main():
    generate_results()
    generate_decorrelated_results()
    
if __name__ == '__main__':
    main()