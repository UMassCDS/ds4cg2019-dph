import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv

def comp_pca(data):
    determinant_data = np.array(data)

    #find the number of variables
    num_var = determinant_data.shape[1]

    #calc is the choose the factor loadings
    calc = np.abs(np.sqrt(1/num_var))

    #cov_mat is the covariance matrix of the data
    cov_mat = np.cov(determinant_data.T)

    #perform PCA on the covariance matrix
    pca = PCA()
    transformed_data = pca.fit(cov_mat).transform(cov_mat)

    #proportion of variance present in each component
    eig = pca.explained_variance_
    n = len(np.where(eig>1)[0])
    transformed_data = transformed_data[:n,:n]

    #weights assigned to each pc
    weights = pca.explained_variance_ratio_/np.sum(pca.explained_variance_ratio_)

    #choose only the pc's who satisfies the constraint
    var_load = pca.components_
    var_load[var_load > calc] = 0

    #scores assigned to each component
    factor_scores = weights @ var_load

    #calculate the health score for every town
    health_status = np.array([determinant_data @ factor_scores]).T
    health_status = MinMaxScaler().fit_transform(health_status)
    health_status = health_status.flatten()
    
    scores = [round(x,2) for x in health_status]
    
    return scores

class HealthScores():
    '''
    Get healthscores for all towns
    '''
    def __init__(self, cols_filepath, data_filepath, pca_filepath, loadings_filepath=None, domain_filepath=None, corrmat_filepath = None):
        '''
        Initialize variables
        '''
        self.read_cols = cols_filepath
        self.data = data_filepath
        self.output = pca_filepath
        self.loadings_file = loadings_filepath
        self.domain_file = domain_filepath
        self.corrmat_file = corrmat_filepath

        self.domains = {'built_environment':['no_vehicle_avail_%', 'comm_car_%', 'comm_carpool_%', 'comm_bus_%', 'comm_walk_%',\
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
        self.var_load = None
        self.n = None
        
    def extract_features(self):
        """
        To extract the column headers so we can map PC components to relevant factors
        """
        data = pd.read_csv(self.read_cols, index_col=0)
        # columns = [x for x in enumerate(data.columns, 1)]
        return list(data.columns)

    def extract_towns(self):
        """
        Extract the town names, so that health score can have meaning
        TODO: FIX FOR ALL PCA
        """
        # data = pd.read_csv(self.read_cols, index_col=0)
        # TODO: clean
        data = pd.read_csv('data/clean_determinant_std.csv', index_col=0)
        towns = list(data.index)
        return towns

    def calc_pca(self, write=True, dom_data=(None, None)):
        """
        Method to calculate factor scores
        """
        data = None
        if dom_data[0]:
            data = dom_data[1]
        else:
            data = pd.read_csv(self.data, index_col=0)
        determinant_data = np.array(data)
        #find the number of variables
        num_var = determinant_data.shape[1]

        #calc is the choose the factor loadings
        calc = np.abs(np.sqrt(1/num_var))

        #cov_mat is the covariance matrix of the data
        cov_mat = np.cov(determinant_data.T)

        #perform PCA on the covariance matrix
        pca = PCA()
        transformed_data = pca.fit(cov_mat).transform(cov_mat)

        #proportion of variance present in each component
        eig = pca.explained_variance_
        self.n = len(np.where(eig>1)[0])
        transformed_data = transformed_data[:self.n,:self.n]

        #weights assigned to each pc
        weights = pca.explained_variance_ratio_/np.sum(pca.explained_variance_ratio_)

        #choose only the pc's who satisfies the constraint
        self.var_load = pca.components_
        self.var_load[self.var_load > calc] = 0

        #scores assigned to each component
        factor_scores = weights @ self.var_load

        #calculate the health score for every town
        health_status = np.array([determinant_data @ factor_scores]).T
        health_status = MinMaxScaler().fit_transform(health_status)
        health_status = health_status.flatten()
        
        scores = [round(x,2) for x in health_status]
        towns = self.extract_towns()

        if(write == True):
            #assign health scores
            health_scores = sorted(zip(towns, scores), key = lambda x: x[1])
            print(health_scores)

            with open(self.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(health_scores)
        return scores 

    def load_data(self):
        data = pd.read_csv(self.data, index_col=0)
        return np.array(data)
    
    def calc_corr_mat(self, A):
        return np.cov(A.T)

    def calc_loadings(self):
        
        #print table of indicator variable loadings
        load_table = []
        for i in range(self.n):
            load_table.append(self.var_load[:, i]) 

        features = self.extract_features()
        loadings = zip(features, *load_table)
        with open(self.loadings_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(loadings)
    
    def score_per_domain(self):
        columns = self.extract_features()
        domains_by_no = {}
        domains = self.domains

        for d in domains:
            for indi in domains[d]:
                try:
                    domains_by_no[d].append(str(columns.index(indi)))
                except KeyError:
                    domains_by_no[d] = [str(columns.index(indi))]

        determinant_data = pd.read_csv(self.data, index_col=0)

        domain_scores = []
        for dom in domains_by_no:
            domain_data = determinant_data[domains_by_no[dom]]
            domain_scores.append(self.calc_pca(write=False, dom_data=(True, domain_data)))

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

def main():
    
    health_obj = HealthScores(cols_filepath="data/clean_determinant_std.csv", data_filepath="data/determinant_data_std.csv",\
       pca_filepath = "output/pca_determinant_std.csv", loadings_filepath="output/loadings_determinant_std.csv", domain_filepath="output/pca_domains_determinant_std.csv",\
           corrmat_filepath = "output/correlation_matrix_determinant_std.csv")
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.score_per_domain()
    health_obj.write_corr_mat()
    
    health_obj = HealthScores(cols_filepath="data/clean_determinant_mn.csv", data_filepath="data/determinant_data_mn.csv",\
       pca_filepath = "output/pca_determinant_mn.csv", loadings_filepath="output/loadings_determinant_mn.csv", domain_filepath="output/pca_domains_determinant_mn.csv")
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.score_per_domain()
    
    health_obj = HealthScores(cols_filepath="data/health_outcomes_std.csv", data_filepath="data/outcome_data_std.csv",\
       pca_filepath = "output/pca_outcome_std.csv", loadings_filepath="output/loadings_outcome_std.csv", corrmat_filepath = "output/correlation_matrix_outcome_std.csv")
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.write_corr_mat()
    
    health_obj = HealthScores(cols_filepath="data/health_outcomes_mn.csv", data_filepath="data/outcome_data_mn.csv",\
       pca_filepath = "output/pca_outcome_mn.csv", loadings_filepath="output/loadings_outcome_mn.csv")
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()    
    
    health_obj = HealthScores(cols_filepath="data/all_data_std.csv", data_filepath="data/all_data_std.csv",\
       pca_filepath = "output/pca_all_std.csv", loadings_filepath="output/loadings_all_std.csv", corrmat_filepath = "output/correlation_matrix_all_std.csv")
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    health_obj.write_corr_mat()

    health_obj = HealthScores(cols_filepath="data/all_data_mn.csv", data_filepath="data/all_data_mn.csv",\
       pca_filepath = "output/pca_all_mn.csv", loadings_filepath="output/loadings_all_mn.csv")
    health_obj.calc_pca(write=True)
    health_obj.calc_loadings()
    
if __name__ == '__main__':
    main()