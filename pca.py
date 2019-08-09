import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import csv
from scipy.stats import pearsonr
from scipy.misc import logsumexp
import collections
from factor_analyzer import FactorAnalyzer

"""
Define global variables that breaks down indicators into domains. 
Keys: domain names
Values: feature names 
Invoked when the dataset has to be split into domains and each domain is handled separately.
"""
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
    HealthScores class creates an object of type healthscore. Contains functions for all Principal Component Analysis parts, 
    and factor analysis as well as correlation analysis. 
    Attributes:
        read_cols:  File to read the column names 
        data: File to read the input data
        output: File to store the output health scores
        loadings_file: File to output the feature scores (factor scores in the paper)
        domain_file: File to write the scores by domain 
        corrmat_file: File to write the correlation matrix
        pvalue_file: File to write the p value matrix
        fa_file: File to write the factor analysis outputs 
        sigcorr_file: File to write the significant correlations 
        decorrelated_file: File to write the decorrelated data
        VER: stores the version of every file (std/mc)
        DC: flag to show if functions are performed on correlated/decorrelated data
        domains: dictionary holding the domain data 
        var_load: hold significant eigenvector correlations from pca 
        n: chosen number of principal components
        pca: represents the output of pca (follows the implementation from the sklearn library: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    '''
    def __init__(self, INFO):
        '''
        Initialize variables when an object of class HealthScores is created. 
        '''
        self.read_cols = INFO['cols_filepath']
        self.data = INFO['data_filepath']
        self.output = INFO['pca_filepath']
        self.loadings_file = INFO['loadings_filepath']
        self.domain_file = INFO['domain_filepath']
        self.corrmat_file = INFO['corrmat_filepath']
        self.pvalue_file = INFO['pvalue_filepath']
        self.fa_file = INFO['fa_filepath']
        self.sigcorr_file = INFO['sigcorr_filepath']
        self.decorrelated_file = INFO['decorrelated_filepath']
        self.VER = INFO['VER']
        self.DC = INFO['DC']
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
        To extract the column headers so we can map PC components to relevant features 
        Returns: 
            List of column names 
        """
        data = pd.read_csv(self.read_cols, index_col=0)
        return list(data.columns)

    def extract_towns(self):
        """
        Extract the town names from the data file. (Allows easy conversion from named index to numbered index)
        Returns:
            List of town names in alphabetical order 
        """
        data = pd.read_csv('data/health_determinants.csv', index_col=0)
        towns = list(data.index)
        return towns

    def calc_pca(self, write=True, dom_data=(None, None), explain_fa_dfilepath=None):
        """
        Method to perform Principal Component Analysis. Steps performed in this function are referenced from the following source - 
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5065523/
        Args: 
            write: Boolean variable to change whether output is written to a file or not (True: writes output to file)
            dom_data(None, None): contains the flag and data that is passed, if PCA is being performed on each domain 
                dom_data[0]: Flag to show if domains are present 
                dom_data[1]: Contains domain data subset 
            explain_fa_filepath: contains file path for writing factor analysis results per domain 
        Returns: 
            scores: list of scores for towns in alphabetical order 
        """
        # Check if the function is acting on domain data, if True, then set Kaiser condition cutoff to 0.01, 
        # else the cutoff is 1.0 
        data = None
        cut_off = None
        if dom_data[0]:
            data = dom_data[1]
            cut_off = 1e-2
        else:
            data = pd.read_csv(self.data, index_col=0)
            cut_off = 1.0

        # Convert data to numpy array for further processing 
        loaded_data = np.array(data)

        #find the number of variables
        num_var = loaded_data.shape[1]

        #calc is the cutoff to find which eigenvector correlations are significant 
        calc = np.abs(np.sqrt(1/num_var))

        #cov_mat is the covariance matrix of the data
        cov_mat = np.cov(loaded_data.T)
        
        #perform PCA on the covariance matrix
        self.pca = PCA()
        self.pca.fit(cov_mat)
        
        # kaiser criterion : Components with eigen_values > cut_off should be retained
        selected_components = np.argwhere(self.pca.explained_variance_>cut_off).flatten()
        selected_vr = self.pca.explained_variance_ratio_[selected_components]

        # second criterion : Among selected components, retain those with proportion of variance  > 10%
        mod_idxs = np.argwhere(selected_vr.flatten()>0.1)
        selected_vr = selected_vr[selected_vr>0.1]
        selected_components = selected_components[mod_idxs].flatten()

        # Check number of selected PCs 
        # If selected PCs are less than 1 because both of the above conditions aren't met, change PCs to 1
        self.n = len(selected_components)
        if self.n < 1:
            self.n = 1

        # Weights assigned to every PC 
        weights = np.exp((np.log(self.pca.explained_variance_) - np.log(logsumexp(self.pca.explained_variance_[:self.n]))))

        # Choose only n selected PCs
        weights = weights[:self.n]

        # Keep only the eigenvector correlations that satisfy the indicator variable loading cutoff 
        self.var_load = self.pca.components_.T[:, :self.n]
        self.var_load[self.var_load > calc] = 0

        # Calculate feature scores for every feature 
        factor_scores = self.var_load @ weights

        # Calculate the health score for every town by multiplying the original data value to the feature score.
        # The final health score for every town is a linear combination of the health scores 
        health_status = np.array([loaded_data @ factor_scores]).T
        health_status = MinMaxScaler().fit_transform(health_status)
        health_status = health_status.flatten()
        
        # Round scores to 2 decimal places
        scores = [round(x,2) for x in health_status]

        # Get town names
        towns = self.extract_towns()

        if(write == True):
            # Combine the towns, health scores to write to a file
            health_scores = sorted(zip(towns, scores), key = lambda x: x[1])

            with open(self.output, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['Town', 'Health Score'])
                writer.writerows(health_scores)
        
        return scores 

    def factor_analysis(self, write=False):
        '''
        Function to get the features and their importance in factor analysis
        Args: 
            write: Boolean variable to choose whether to write the output to a file or not. 
        Returns: 
            inp: pandas DataFrame that contains the original data 
            sorted_mag: list of tuples to store the features and their importance in decreasing format
        '''
        inp = pd.read_csv(self.data, index_col=0)

        # fa stores the output of the factor_analyzer 
        fa = FactorAnalyzer(n_factors = self.n, rotation='varimax')
        
        # fits the input to get feature importances
        fa.fit(inp)

        # gets the factor loadings 
        magnitude = fa.get_communalities()

        feat = self.extract_features()

        # Dictionary to hold the correct feature name to the number
        mag_dict = {}
        for t,f in enumerate(feat):
            mag_dict[f] = magnitude[t]
        
        sorted_mag = sorted(mag_dict.items(), key=lambda kv:kv[1], reverse=True)
        
        # Writes the output of factor analysis to a file
        if write==True:
            factors = pd.DataFrame(sorted_mag, columns=['Feature','Importance'])
            factors.to_csv(self.fa_file, index=False)
        
        return inp, sorted_mag

    def factor_analysis_perdomain(self, write=False):
        '''
        Function to perform factor analysis per domain
        Args: 
            write: Boolean variable to choose whether to write the output to a file or not. 
        Returns:
            None
        '''
        # load domains into domain data 
        domain_data = self.load_domains()

        # Get mapping of original numbers to shuffled numbers 
        true_num = list(pd.read_csv(self.data, index_col=0))
        for d in self.domains:
            curr = domain_data[d]
            col = list(curr)
            ind = []
            for c in col:
                ind.append(true_num.index(c))
            index_num = list(map(int, ind))
            column_name = list(np.array(self.extract_features())[index_num])

            inp = domain_data[d]
            fa = FactorAnalyzer(n_factors = self.n, rotation='varimax')

            # In some cases, factor analysis does not success 
            try:
                fa.fit(inp)
            except:
                print('Data from '+str(d)+' domain cannot be factorized as it results in a singular matrix.')
                continue
            magnitude = fa.get_communalities()

            mag_dict={}
            for i,_ in enumerate(column_name):
                mag_dict[column_name[i]] = magnitude[i]

            sorted_mag = sorted(mag_dict.items(), key=lambda kv:kv[1], reverse=True)

            if write==True:
                factors = pd.DataFrame(sorted_mag, columns=['Feature','Importance'])
                if self.DC:
                    factors.to_csv('output/fa_decorrelated_'+str(d)+'_'+str(self.VER)+'.csv', index=False)
                else:
                    factors.to_csv('output/fa_'+str(d)+'_'+str(self.VER)+'.csv', index=False)

    def load_data(self):
        '''
        Read data from file and convert to numpy array
        Args:
            None
        Returns:
            Numpy array of data 
        '''
        data = pd.read_csv(self.data, index_col=0)
        return np.array(data)
    
    def calc_cov_mat(self, A):
        '''
        Calculate covariance matrix of given data
        Args:
            A: data matrix
        Returns:
            covariance matrix of A 
        '''
        return np.cov(A.T)
    
    def calc_corr_mat(self, A):
        '''
        Calculate correlation matrix of given data
        Args:
            A: data matrix
        Returns:
            correlation matrix of A 
        '''
        return np.corrcoef(A, rowvar = False)

    def calc_loadings(self): 
        '''
        Print and calculate the eigenvector correlations that account 
        for feature scores in the health score calculations
        Args:
            None
        Returns:
            None
        '''
        # Extracts the loadings from the PCA calculation in calc_pca
        load_table = []
        for i in range(self.n):
            load_table.append(self.var_load[:, i])
        features = np.array([self.extract_features()]).T
        load_table = np.array([*load_table]).T
        
        pc_headers = ['Feature']
        for i in range(0,self.n):
            pc_headers.append('pc' + str(i+1))

        load_table = np.concatenate((features, load_table), axis = 1)

        # write to file
        df = pd.DataFrame(data=load_table, columns = pc_headers)
        df.set_index('Feature',inplace=True)
        df.to_csv(self.loadings_file)
    
    def load_domains(self):
        '''
        Function to load domain data from the entire dataset. Reads the GLOBAL DOMAINS keys to extract domain specific data 
        from the full dataset. 
        Args:
            None
        Returns:
            domain_data: Dictionary containing domain data 
        '''
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
        '''
        Function to calculate the health score by running PCA on each individual domain. 
        Writes the output of PCA per domain to the domain specific file 
        Args:
            None
        Returns:
            None
        '''
        #Extract columns, original full dataset and the reference feature number 
        columns = self.extract_features()
        domains_by_no = {}
        determinant_data = pd.read_csv(self.data, index_col=0)
        true_num = list(determinant_data)

        # Map every feature to domain
        for d in self.domains:
            for indi in self.domains[d]:
                try:
                    domains_by_no[d].append(str(true_num[columns.index(indi)]))
                except KeyError:
                    domains_by_no[d] = [str(true_num[columns.index(indi)])]

        # Call on calc_pca to return the scores for each individual domain
        domain_scores = []
        for dom in domains_by_no:
            domain_data = determinant_data[domains_by_no[dom]]
            if domain_data.shape[1] > 1:
                domain_scores.append(self.calc_pca(write=False, dom_data=(dom, domain_data), explain_fa_dfilepath='output/fa_'+str(dom) + '_' + self.VER + '.csv'))
            else:
                domain_scores.append([round(x,2) for x in MinMaxScaler().fit_transform(np.array(domain_data)).flatten()])
        
        # Round the health scores per domain and write to file 
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
        '''
        Function to write the correlation matrix to file. 
        Args:
            None
        Returns: 
            None
        '''
        # Uses the calc_corr_mat function to calculate the correlation matrix of the data
        A = self.calc_corr_mat(self.load_data())
        # Extracts and maps correct features to the columns
        feat = np.array([self.extract_features()]).T
        # combines correlation data with the feature names
        A = np.concatenate((feat, A), axis = 1)
        headers = ["\\"] + self.extract_features()
        
        # Writes correlation matrix to file 
        df = pd.DataFrame(data = A, columns = headers)
        df.set_index("\\", inplace = True)
        
        df.to_csv(self.corrmat_file)
    
    def write_corr_mat_per_dom(self):
        '''
        Function to write the correlation matrix per domain to domain specific files. 
        Args:
            None
        Returns: 
            None
        '''
        # Extract domain specific data
        data = self.load_domains()
        # Retain the original column ordering 
        true_num = list(pd.read_csv(self.data, index_col=0))

        # For every domain, use calc_corr_mat to do correlation analysis and write to file 
        for d in self.domains:
            curr = data[d]
            col = list(curr)
            ind = []
            for c in col:
                ind.append(true_num.index(c))
            index_num = list(map(int, ind))
            column_name = list(np.array(self.extract_features())[index_num])
            index = np.array([column_name]).T
            matrix = self.calc_corr_mat(curr)
            try:
                matrix = np.concatenate((index, matrix), axis =1)
            except ValueError:
                matrix = np.array([np.array([self.calc_corr_mat(curr)])])
                matrix = np.concatenate((index, matrix), axis =1)
            headers = ["\\"] + column_name
            df = pd.DataFrame(data = matrix, columns = headers)
            df.set_index("\\", inplace=True)
            if self.DC:
                df.to_csv('output/correlation_matrix_decorrelated_' + d + '.csv')
            else:
                df.to_csv('output/correlation_matrix_' + d + '.csv')
    
    def p_values(self, data):
        '''
        Function to calculate p values for the data matrix 
        Args:
            data: data array
        Returns: 
            numpy array of p values
        '''
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
        '''
        Function to write the p value matrix to specific file
        Args:
            None
        Returns:
            None
        '''
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
        '''
        Function to write the domain specific p value matrix to domain file
        Args:
            None
        Returns:
            None
        '''
        data = self.load_domains()
        true_num = list(pd.read_csv(self.data, index_col=0))
        for d in self.domains:
            curr = data[d]
            col = list(curr)
            ind = []
            for c in col:
                ind.append(true_num.index(c))
            index_num = list(map(int, ind))
            column_name = list(np.array(self.extract_features())[index_num])
            index = np.array([column_name]).T

            pvals = self.p_values(np.array(curr))
            headers = ["\\"] + column_name
            pvals = np.concatenate((index, pvals), axis = 1)

            df = pd.DataFrame(data = pvals, columns = headers)
            df.set_index("\\", inplace = True)
            if self.DC:
                df.to_csv('output/p_values_decorrelated_'+d+'.csv')
            else:
                df.to_csv('output/p_values_'+d+'.csv')
    
    def write_significant_correlations(self, cm_file, pval_file, write_file):
        '''
        Function to write significant correlations to separate file
        Significant correlations are those that have a r**2 > 0.8 and pval < 0.05
        The rest of correlations are made to be zero, so only significant correlations are > 0.0
        Args:
            cm_file: correlation matrix file
            pval_file: p value file
            write_file: output significant correlations file
        Returns:
            None
        '''
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
    
    def correlation_analysis(self):
        '''
        Function to drop signficant correlations from the original data and form decorrelated data sets. 
        Reads from the significant correlations file and drops all the columns (from least important features on factor analysis)
        The output data is written into data folder as decorrelated output
        Args:
            None
        Returns:
            None 
        Prints the names of the columns that were dropped in every subset of PCA calculations
        '''
        inp, sorted_mag = self.factor_analysis()
        sig_corr = pd.read_csv(self.sigcorr_file, index_col=0)

        num_cov = {}
        for t,f in enumerate(self.extract_features()):
            num_cov[f] = str(t)

        columns_to_drop = []

        for item in sorted_mag:
            if num_cov[item[0]] not in columns_to_drop:
                col = np.array(sig_corr[item[0]])
                location = list(np.where(col != 0))[0]
                for loc in location:
                    if str(loc) not in columns_to_drop:
                        columns_to_drop.append(str(loc))

        dropped_col_names = []
        name_to_num = {}
        for t,f in enumerate(self.extract_features()):
            name_to_num[t] = f
        for col in columns_to_drop:
            dropped_col_names.append(name_to_num[int(col)])
        
        print('Dropped column names in ',self.data,' is: ', dropped_col_names)
        decorrelated = inp.drop(columns=columns_to_drop)
        column_num = list(map(int, list(decorrelated)))
        column_name = list(np.array(self.extract_features())[column_num])
        decorrelated_names = decorrelated.set_axis(column_name, axis = 1, inplace=False)
        decorrelated_names.set_axis(self.extract_towns(), axis = 0, inplace=True)
        decorrelated.to_csv(self.decorrelated_file + '.csv')
        decorrelated_names.to_csv(self.decorrelated_file + '_columns.csv')