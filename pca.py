import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv
from scipy.stats import pearsonr
from scipy.misc import logsumexp
import collections
from factor_analyzer import FactorAnalyzer

GLOBAL_DOMAINS = {'built_environment':['no_vehicle_avail_%', 'comm_car_%', 'comm_carpool_%', 'comm_bus_%', 'comm_walk_%',\
                    'comm_cycle_%', 'comm_taxi_%', 'comm_wfh_%','tobbaco_retailers_2019_%', 'liquor_per1000', 'supermarket_per1000', 
                    'food_est_per1000'], 
                    'community_context':['race_Wh_%', 'race_Afam_%', 'race_Alaska_%', 'race_Asian_%', 'race_Hawaii_%', \
                    'race_Other_%', 'race_Two_%', 'race_Hisp_%', 'age_lt18_%', 'young_females_%', 'median_age', 'lt5years_%', 
                    'population_density', 'rural_population_%', 'english_verywell_%', 'print_media_circ_%'],
                    'economy':['below_poverty_level_%', 'debt_pervaluation_%', 
                    'households_recfoodstamps_%', 'snap_%', 'median_household_income_USD', 'income_capita_USD', 'gini_index', 'median_earnings_retail_USD', 'poverty_child_%'],
                    'education':['exp_perpupil_USD', 'preschool_%', 'students_disabilities_%', 'students_highneeds_%', 'student_teacher_ratio', 
                    'libraries_%','exemplary_%', 'proficient_%', 'need_improve_%', 'unsatis_%', 'read_write', 'math', 'dropout_%', 'college_%', 
                    'avg_ela', 'avg_math','less_than_highschool_%', 'highschool_grad_%', 'associates_degree_%', 'bachelors_degree_%', 
                    'public_school_%', 'private_school_%','lep_%', 'students_econodisadv_%'],
                    'employment':['unemployment_rate', 'labor_force_rate', 'emp_rate', 'emp_business_%', 'emp_computer_%', 'emp_legal_%', 
                    'emp_healthcare_%', 'emp_healthsupport_%', 'emp_protective_%', 'emp_foodprepare_%', 'emp_clean_%', 'emp_personalcare_%', 
                    'emp_sales_%', 'emp_admin_%', 'emp_farm_%', 'emp_construct_%', 'emp_repair_%', 'emp_production_%', 'emp_transport_%', 
                    'emp_material_%', 'emp_pop_ratio_wh_only', 'emp_pop_ratio_hisp', 'employed_male_%', 'employed_female_%'],
                    'healthcare_access':['licensed_nurses_per1000', 'dentists_per1000', 'physician_assisted_%', 'pa_provider_density_per10000', 
                    'physicians_per1000', 'pcp_per1000', 'psychiatrist_per1000', 'psychologist_per1000', 'pharmacist_per1000', 
                    'registerednurses_per1000', 'advancedprn_per1000', 'health_insurance_coverage_%'],
                    'housing':['house_noplumbing_%', 'house_nokitchen_%', 'house_lt1939_%', 'house_1940to1999_%', 'house_2000to2014_%',
                    'median_owner_home_USD', 'median_sale_homes_USD', 'house_income_gt30_USD_%', 'gross_rent_%', 'property_val_USD', 
                    'costas%income_wmgage_lt20%_%', 'costas%income_wmgage_20to25%_%', 'costas%income_wmgage_25to30%_%', 'costas%income_wmgage_30to35%_%',
                    'costas%income_wmgage_gt35%_%', 'costas%income_womgage_lt10%_%', 'costas%income_womgage_10to15%_%', 'costas%income_womgage_15to20%_%', 
                    'costas%income_womgage_20to25%_%', 'costas%income_womgage_25to30%_%', 'costas%income_womgage_30to35%_%','costas%income_womgage_gt35%_%',
                    'homeless_Shelters_per_1000', 'moved_lastyear_%', 'renter_occupied_%', 'vacant_rentals_%', 'owner_homes_%', 'occ_lt0.5_%', 
                    'occ_0.5to1.0_%', 'occ_1.01to1.5_%', 'occ_1.51to2.0_%', 'occ_gt2.01_%'],
                    'violence':['crimes_against_persons_%', 'crimes_against_property_%', 'crimes_against_society_%']}

class HealthScores():
    '''
    Get healthscores for all towns
    '''
    def __init__(self, INFO):
        '''
        Initialize variables
        '''
        self.read_cols = INFO['cols_filepath']
        self.data = INFO['data_filepath']
        self.output = INFO['pca_filepath']
        self.loadings_file = INFO['loadings_filepath']
        self.domain_file = INFO['domain_filepath']
        self.corrmat_file = INFO['corrmat_filepath']
        self.pvalue_file = INFO['pvalue_filepath']
        self.var_file = INFO['variance_filepath']
        self.sigcorr_file = INFO['sigcorr_filepath']
        self.decorrelated_file = INFO['decorrelated_filepath']
        self.VER = INFO['VER']
        self.domains = {}

        if INFO['domain_filepath']:
            feat = self.extract_features()
            for d in GLOBAL_DOMAINS:
                for indi in GLOBAL_DOMAINS[d]:
                    if indi in feat:
                        try:
                            self.domains[d].append(indi)
                        except KeyError:
                            self.domains[d] = [indi]

        self.var_load = None
        self.n = None
        self.pca = None
        
    def extract_features(self):
        """
        To extract the column headers so we can map PC components to relevant factors
        """
        data = pd.read_csv(self.read_cols, index_col=0)
        # columns = [x for x in enumerate(data.columns)]
        return list(data.columns)

    def extract_towns(self):
        """
        Extract the town names, so that health score can have meaning
        TODO: FIX FOR ALL PCA
        """
        # data = pd.read_csv(self.read_cols, index_col=0)
        # TODO: clean
        data = pd.read_csv('data/health_determinants.csv', index_col=0)
        towns = list(data.index)
        return towns

    def calc_pca(self, write=True, dom_data=(None, None), explain_var_dfilepath=None):
        """
        Method to calculate factor scores
        """
        data = None
        cut_off = None
        if dom_data[0]:
            data = dom_data[1]
            cut_off = 1e-2
        else:
            data = pd.read_csv(self.data, index_col=0)
            cut_off = 1.0
        determinant_data = np.array(data)
        #find the number of variables
        num_var = determinant_data.shape[1]

        #calc is the choose the factor loadings
        calc = np.abs(np.sqrt(1/num_var))

        #cov_mat is the covariance matrix of the data
        cov_mat = np.cov(determinant_data.T)
        
        #perform PCA on the covariance matrix
        self.pca = PCA()
        self.pca.fit(cov_mat)

        # kaiser criterion : Components with eigen_values > 1.0 should be retained
        selected_components = np.argwhere(self.pca.explained_variance_>cut_off).flatten()
        selected_vr = self.pca.explained_variance_ratio_[selected_components]

        # second criterion : Among selected components, retain those with proportion of variance  > 10%
        mod_idxs = np.argwhere(selected_vr.flatten()>0.1)
        selected_vr = selected_vr[selected_vr>0.1]
        selected_components = selected_components[mod_idxs].flatten()

        self.n = len(selected_components)

        #Print explained variance
        #if explain_var_dfilepath != None:
        #    self.explain_var(pca, per_dom_filepath=explain_var_dfilepath, domain=dom_data[0])
        #else:
        #    self.explain_var(pca)

        #weights assigned to each pc
        weights = np.exp((np.log(self.pca.explained_variance_) - np.log(logsumexp(self.pca.explained_variance_[:self.n]))))

        weights = weights[:self.n]

        #choose only the pc's who satisfies the constraint
        self.var_load = self.pca.components_.T[:, :self.n]
        self.var_load[self.var_load > calc] = 0

        #scores assigned to each component
        factor_scores = self.var_load @ weights

        #calculate the health score for every town
        health_status = np.array([determinant_data @ factor_scores]).T
        health_status = MinMaxScaler().fit_transform(health_status)
        health_status = health_status.flatten()
        
        scores = [round(x,2) for x in health_status]
        towns = self.extract_towns()

        if(write == True):
            #assign health scores
            health_scores = sorted(zip(towns, scores), key = lambda x: x[1])

            with open(self.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['Town', 'Health Score'])
                writer.writerows(health_scores)
        
        return scores 

    def load_data(self):
        data = pd.read_csv(self.data, index_col=0)
        return np.array(data)
    
    def calc_cov_mat(self, A):
        return np.cov(A.T)
    
    def calc_corr_mat(self, A):
        return np.corrcoef(A, rowvar = False)

    def calc_loadings(self): 
        #print table of indicator variable loadings
        load_table = []
        for i in range(self.n):
            load_table.append(self.var_load[:, i])
        features = np.array([self.extract_features()]).T
        load_table = np.array([*load_table]).T
        
        pc_headers = ['Feature']
        for i in range(0,self.n):
            pc_headers.append('pc' + str(i+1))

        load_table = np.concatenate((features, load_table), axis = 1)

        df = pd.DataFrame(data=load_table, columns = pc_headers)
        df.set_index('Feature',inplace=True)
        df.to_csv(self.loadings_file)
    
    def load_domains(self):
        columns = self.extract_features()
        domains_by_no = {}
        determinant_data = pd.read_csv(self.data, index_col=0)
        true_num = list(determinant_data)

        for d in self.domains:
            for indi in self.domains[d]:
                try:
                    domains_by_no[d].append(str(true_num[columns.index(indi)]))
                except KeyError:
                    domains_by_no[d] = [str(true_num[columns.index(indi)])]

        domain_data = {}
        for dom in domains_by_no:
            domain_data[dom] = determinant_data[domains_by_no[dom]]
        
        return domain_data
    
    def score_per_domain(self):
        columns = self.extract_features()
        domains_by_no = {}
        determinant_data = pd.read_csv(self.data, index_col=0)
        true_num = list(determinant_data)

        for d in self.domains:
            for indi in self.domains[d]:
                try:
                    domains_by_no[d].append(str(true_num[columns.index(indi)]))
                except KeyError:
                    domains_by_no[d] = [str(true_num[columns.index(indi)])]

        domain_scores = []
        for dom in domains_by_no:
            domain_data = determinant_data[domains_by_no[dom]]
            if domain_data.shape[1] > 1:
                domain_scores.append(self.calc_pca(write=False, dom_data=(dom, domain_data), explain_var_dfilepath='output/variance_'+str(dom) + '_' + self.VER + '.csv'))
            else:
                domain_scores.append([round(x,2) for x in MinMaxScaler().fit_transform(np.array(domain_data)).flatten()])
        domain_scores = np.array(domain_scores).T
        avg_dom = np.array([[round(x,2) for x in np.average(domain_scores, axis =1)]]).T
        domain_scores = np.concatenate((domain_scores, avg_dom), axis = 1)
        towns = self.extract_towns()
        dom_names = ['index','BE', 'CC', 'ECON', 'EDU', 'EMP', 'HEA', 'HOU', 'VIO', 'AVG']
        towns = np.array([towns]).T
        domain_scores = np.concatenate((towns, domain_scores), axis = 1)

        df = pd.DataFrame(data = domain_scores, columns = dom_names)
        df.set_index('index', inplace=True)

        df.to_csv(self.domain_file)

    def write_corr_mat(self):
        A = self.calc_corr_mat(self.load_data())
        feat = np.array([self.extract_features()]).T
        A = np.concatenate((feat, A), axis = 1)
        headers = ["\\"] + self.extract_features()
        
        df = pd.DataFrame(data = A, columns = headers)
        df.set_index("\\", inplace = True)
        
        df.to_csv(self.corrmat_file)
    
    def write_corr_mat_per_dom(self):
        data = self.load_domains()
        for d in self.domains:
            curr = data[d]
            col = list(curr)
            column_num = list(map(int, col))
            column_name = list(np.array(self.extract_features())[column_num])
            index = np.array([column_name]).T
            matrix = self.calc_corr_mat(curr)
            matrix = np.concatenate((index, matrix), axis =1)
            headers = ["\\"] + column_name
            df = pd.DataFrame(data = matrix, columns = headers)
            df.set_index("\\", inplace=True)
            df.to_csv('output/correlation_matrix_' + d + '.csv')
    
    def p_values(self, data):
        _,N = data.shape
        pvals = []
        for i in range(N):
            row = []
            for j in range(N):
                _, p = pearsonr(data[:, i], data[:, j])
                row.append(p)
            pvals.append(row)
        return np.array(pvals)
    
    def write_p_values(self):
        data = self.load_data()
        pvals = self.p_values(data)
        feat = self.extract_features()
        index = np.array([feat]).T
        pvals = np.concatenate((index, pvals), axis = 1)
        headers = ["\\"] + feat
        df = pd.DataFrame(data= pvals, columns= headers)
        df.set_index("\\", inplace=True)
        df.to_csv(self.pvalue_file)
    
    def write_p_values_per_dom(self):
        data = self.load_domains()
        for d in self.domains:
            curr = data[d]
            col = list(curr)
            column_num = list(map(int, col))
            column_name = list(np.array(self.extract_features())[column_num])
            index = np.array([column_name]).T

            pvals = self.p_values(np.array(curr))
            headers = ["\\"] + column_name
            pvals = np.concatenate((index, pvals), axis = 1)

            df = pd.DataFrame(data = pvals, columns = headers)
            df.set_index("\\", inplace = True)
            df.to_csv('output/p_values_'+d+'.csv')
    
    def write_significant_correlations(self, cm_file, pval_file, write_file):
        cm = pd.read_csv(cm_file, index_col = 0)
        feat = list(cm)
        cm = np.array(cm)
        pval = np.array(pd.read_csv(pval_file, index_col = 0))

        cm[np.abs(cm) < 0.8] = 0
        cm[pval > 0.05] = 0
        cm[cm == 1] = 0

        index = np.array([feat]).T
        headers = ["\\"] + feat
        cm = np.concatenate((index, cm), axis = 1)

        df = pd.DataFrame(data = cm, columns = headers)
        df.set_index("\\", inplace = True)
        df.to_csv(write_file)

    def explain_var(self, pca, per_dom_filepath=None, domain = None):
        var = pca.explained_variance_ratio_
        if per_dom_filepath==None:
            features = self.extract_features()
            variances = list(zip(features, var))
            with open(self.var_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['Feature', 'Eigen value (%)'])
                writer.writerows(variances)
        else:
            features = self.domains[domain]
            variances = list(zip(features, var))
            with open(per_dom_filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['Feature', 'Eigen value (%)'])
                writer.writerows(variances)
    
    def correlation_analysis(self):
        fa = FactorAnalyzer(n_factors = self.n, rotation='varimax')
        inp = pd.read_csv(self.data, index_col=0)
        fa.fit(inp)
        magnitude = fa.get_communalities()

        feat = self.extract_features()
        mag_dict = {}
        for t,f in enumerate(feat):
            mag_dict[f] = magnitude[t]
        
        sorted_mag = sorted(mag_dict.items(), key=lambda kv:kv[1], reverse=True)
        
        sig_corr = pd.read_csv(self.sigcorr_file, index_col=0)

        columns_to_drop = []

        for item in sorted_mag:
            if item[0] not in columns_to_drop:
                col = np.array(sig_corr[item[0]])
                location = list(np.where(col != 0))[0]
                for loc in location:
                    if loc not in columns_to_drop:
                        columns_to_drop.append(str(loc))
        decorrelated = inp.drop(columns=columns_to_drop)
        column_num = list(map(int, list(decorrelated)))
        column_name = list(np.array(self.extract_features())[column_num])
        decorrelated_names = decorrelated.set_axis(column_name, axis = 1, inplace=False)
        decorrelated_names.set_axis(self.extract_towns(), axis = 0, inplace=True)
        decorrelated.to_csv(self.decorrelated_file + '.csv')
        decorrelated_names.to_csv(self.decorrelated_file + '_columns.csv')


DETERMINANT_STD = {'cols_filepath':"data/health_determinants.csv", 
                    'data_filepath':"data/determinant_data_std.csv",
                    'pca_filepath':"output/pca_determinant_std.csv", 
                    'loadings_filepath':"output/loadings_determinant_std.csv",
                    'domain_filepath':"output/pca_domains_std.csv",
                    'corrmat_filepath':"output/correlation_matrix_determinant.csv", 
                    'pvalue_filepath':"output/p_values_determinant.csv", 
                    'variance_filepath':"output/variance_determinant_std.csv",
                    'sigcorr_filepath':"output/significant_correlations_determinant.csv",
                    'decorrelated_filepath':"data/decorrelated_determinant_data_std",
                    'VER':'std'}

DETERMINANT_MN = {'cols_filepath':"data/health_determinants.csv",
                    'data_filepath':"data/determinant_data_mn.csv",
                    'pca_filepath':"output/pca_determinant_mn.csv", 
                    'loadings_filepath':"output/loadings_determinant_mn.csv", 
                    'domain_filepath':"output/pca_domains_mn.csv",
                    'corrmat_filepath':"output/correlation_matrix_determinant.csv",
                    'pvalue_filepath':"output/p_values_determinant.csv",
                    'variance_filepath':"output/variance_determinant_mn.csv", 
                    'sigcorr_filepath':"output/significant_correlations_determinant.csv",
                    'decorrelated_filepath':'data/decorrelated_determinant_data_mn',
                    'VER':'mn'}

OUTCOME_STD = {'cols_filepath':"data/health_outcomes.csv",
                'data_filepath':"data/outcome_data_std.csv",
                'pca_filepath':"output/pca_outcome_std.csv", 
                'loadings_filepath':"output/loadings_outcome_std.csv",
                'domain_filepath':None,
                'corrmat_filepath':"output/correlation_matrix_outcome.csv", 
                'pvalue_filepath':"output/p_values_outcome.csv",
                'variance_filepath':"output/variance_outcome_std.csv", 
                'sigcorr_filepath':'output/significant_correlations_outcome.csv',
                'decorrelated_filepath':'data/decorrelated_outcome_data_std',
                'VER':'std'}

OUTCOME_MN = {'cols_filepath':"data/health_outcomes.csv", 
                'data_filepath':"data/outcome_data_mn.csv",
                'pca_filepath':"output/pca_outcome_mn.csv",
                'loadings_filepath':"output/loadings_outcome_mn.csv",
                'domain_filepath':None,
                'corrmat_filepath':"output/correlation_matrix_outcome.csv", 
                'pvalue_filepath':"output/p_values_outcome.csv",
                'variance_filepath':"output/variance_outcome_mn.csv",
                'sigcorr_filepath':'output/significant_correlations_outcome.csv',
                'decorrelated_filepath':'data/decorrelated_outcome_data_mn',
                'VER':'mn'}

ALL_STD = {'cols_filepath':"data/all_data.csv",
            'data_filepath':"data/all_data_std.csv",
            'pca_filepath':"output/pca_all_std.csv",
            'loadings_filepath':"output/loadings_all_std.csv",
            'domain_filepath':None,
            'corrmat_filepath':"output/correlation_matrix_all.csv",
            'pvalue_filepath':"output/p_values_all.csv", 
            'variance_filepath':"output/variance_all_std.csv",
            'sigcorr_filepath':"output/significant_correlations_all.csv",
            'decorrelated_filepath':"data/decorrelated_all_data_std",
            'VER':'std'}

ALL_MN = {'cols_filepath':"data/all_data.csv",
            'data_filepath':"data/all_data_mn.csv",
            'pca_filepath':"output/pca_all_mn.csv",
            'loadings_filepath':"output/loadings_all_mn.csv",
            'domain_filepath':None,
            'corrmat_filepath':"output/correlation_matrix_all.csv",
            'pvalue_filepath':"output/p_values_all.csv",
            'variance_filepath':"output/variance_all_mn.csv",
            'sigcorr_filepath':"output/significant_correlations_all.csv", 
            'decorrelated_filepath':"data/decorrelated_all_data_mn",
            'VER':'mn'}

DC_DETERMINANT_STD = {'cols_filepath':"data/decorrelated_determinant_data_std_columns.csv",
                        'data_filepath':"data/decorrelated_determinant_data_std.csv",
                        'pca_filepath':"output/pca_decorrelated_determinant_std.csv",
                        'loadings_filepath':"output/loadings_decorrelated_determinant_std.csv",
                        'domain_filepath':"output/pca_decorrelated_domains_std.csv",
                        'corrmat_filepath':"output/correlation_matrix_decorrelated_determinant.csv",
                        'pvalue_filepath':"output/p_values_decorrelated_determinant.csv",
                        'variance_filepath':None,
                        'sigcorr_filepath':"output/significant_correlations_decorrelated_determinant.csv",
                        'decorrelated_filepath':None,
                        'VER':'std'}

def generate_results():
    health_obj = HealthScores(DETERMINANT_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings() 
    health_obj.score_per_domain()
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_corr_mat_per_dom()
    health_obj.write_p_values_per_dom()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    for d in health_obj.domains:
        health_obj.write_significant_correlations('output/correlation_matrix_'+ d + '.csv', 'output/p_values_'+d+'.csv', 'output/significant_correlations_'+d+'.csv')
    health_obj.correlation_analysis()

    health_obj = HealthScores(DETERMINANT_MN)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.score_per_domain()
    health_obj.correlation_analysis()
    
    health_obj = HealthScores(OUTCOME_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    health_obj.correlation_analysis()
    
    health_obj = HealthScores(OUTCOME_MN)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.correlation_analysis()

    health_obj = HealthScores(ALL_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    health_obj.correlation_analysis()

    health_obj = HealthScores(ALL_MN)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.correlation_analysis()

def generate_decorrelated_results():
    health_obj = HealthScores(DC_DETERMINANT_STD)
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.score_per_domain()
    health_obj.write_corr_mat()
    health_obj.write_p_values()
    health_obj.write_corr_mat_per_dom()
    health_obj.write_p_values_per_dom()
    health_obj.write_significant_correlations(health_obj.corrmat_file, health_obj.pvalue_file, health_obj.sigcorr_file)
    for d in health_obj.domains:
        health_obj.write_significant_correlations('output/correlation_matrix_decorrelated_'+ d + '.csv', 'output/p_values_decorrelated_'+d+'.csv', 'output/significant_correlations_decorrelated_'+d+'.csv')
    
def main():
    #generate_results()
    generate_decorrelated_results()
    
if __name__ == '__main__':
    main()
