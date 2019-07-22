import pandas as pd 
import numpy as np 
from sklearn.decomposition import PCA

domains = {'built_environment':['no_vehicle_avail_%', 'comm_car_%', 'comm_carpool_%', 'comm_bus_%', 'comm_walk_%',\
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

df = pd.read_csv('data/clean_determinants.csv', index_col=0)
columns = list(df)

domains_by_no = {}

for d in domains:
    for indi in domains[d]:
        try:
            domains_by_no[d].append(str(columns.index(indi)))
        except KeyError:
            domains_by_no[d] = [str(columns.index(indi))]

determinant_data = pd.read_csv('data/determinant_data.csv', index_col=0)

for dom in domains_by_no:
    domain_data = determinant_data[domains_by_no[dom]]
    print(domain_data.head())