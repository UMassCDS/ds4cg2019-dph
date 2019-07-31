import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
def nomalize_percent(x):
    return float(x/100)

file_path = '../Aggregated_Data_v3.csv'

df = pd.read_csv(file_path)

inc_data = ['vehicles_#', 'poverty_all_#', 'emp_Pop_Ratio_Wh', 'emp_Pop_Ration_Afam', 'emp_Pop_Ratio_Alaska', 'emp_Pop_Ratio_Asian', 'emp_Pop_Ratio_Hawaii', 'emp_Pop_Ratio_Other', 'emp_Pop_Ratio_Two', 'emp_Pop_Ratio_Hisp']

df = df.drop(inc_data, axis = 1)

household_no = np.array(list(df['Total # of households']))

df = df.drop('Total # of households', axis = 1)

columns = list(df)[:65]

sub_set = df[columns]

pop = np.array(list(df['Total population']))

munip = list(df['Unnamed: 0'])

per_k = pop/1000

emp_rate = np.array(list(df['emp_rate']))/100

sparse = []
clean = []
rate = []
norm = []
perc = []


for c in columns[1:]:
    if (sub_set[c].isna().sum() > 175):
        sub_set = sub_set.drop([c], axis = 1)
        sparse.append(c)
    else:
        name =  sub_set[c].name
        dtype = sub_set[c].dtype
        if(dtype != 'float64'):
            temp = pd.to_numeric(sub_set[c], errors='ignore')
            clean.append(c)
            for i,j in temp.iteritems():
                try:
                    float(j)
                except ValueError:
                    if(j == '-'):
                        temp[i] = np.nan
                    elif name == 'median_sale_homes_USD' or name == 'rent_income_USD':
                        temp[i] = float(j[:-1].replace(',',''))
                    else:
                        print(name)
                        print(j)
                        print(type(j))
            sub_set.loc[: ,c] = pd.to_numeric(temp)
        if c == 'homeless_Shelters':
            rate.append(c)
            sub_set.loc[:, c] = np.array(list(sub_set[c])/per_k)
            sub_set.rename(index=str, columns={c:c+'_per_1000'}, inplace = True)
        if c == 'labor_force_rate' or c == 'emp_rate' or c == 'age_lt18' or c== 'house_lt1939' or c == 'house_1940to1999' or c == 'house_2000to2014' or c == 'poverty_child_%' or c == 'rent_income_USD' or c == 'comm_car_%' or c == 'comm_carpool_%' or c == 'comm_bus_%'	or c == 'comm_walk_%' or c == 'comm_cycle_%' or c == 'comm_taxi_%' or c == 'comm_wfh_%' or c == '% no vehicle is available' or c == '% below poverty level':
            norm.append(c)
            sub_set.loc[:, c] = sub_set[c].apply(nomalize_percent)
            if c == 'age_lt18' or c== 'house_lt1939' or c == 'house_1940to1999' or c == 'house_2000to2014':
                sub_set.rename(index=str, columns={c:c+'_%'}, inplace = True)
            if c == '% no vehicle is available':
                sub_set.rename(index=str, columns={c:'no_vehicle_avail_%'}, inplace = True)
            if c == '% below poverty level':
                sub_set.rename(index=str, columns={c:'below_poverty_level_%'}, inplace = True)
            if c == 'rent_income_USD':
                sub_set.rename(index=str, columns={c:'gross_rent_%'}, inplace = True)
        if c == 'race_Wh' or c == 'race_Afam' or c == 'race_Alaska' or c == 'race_Asian' or c == 'race_Hawaii' or c == 'race_Other' or c == 'race_Two' or c == 'race_Hisp' or c == 'eng' or c == 'owner_homes_#' or c == 'moved_lastyear_#':
            perc.append(c)
            sub_set.loc[:, c] = np.array(list(sub_set[c])/pop)
            replace = ''
            if c == 'owner_homes_#' or c == 'moved_lastyear_#':
                replace = c.replace('#', '%')
            else:
                replace = c + '_%'
            sub_set.rename(index=str, columns={c:replace}, inplace = True)
        if c == 'emp_business' or c == 'emp_computer' or c == 'emp_legal' or c == 'emp_healthcare' or c == 'emp_healthsupport' or c == 'emp_protective' or c == 'emp_foodprepare' or c == 'emp_clean' or c == 'emp_personalcare' or c == 'emp_sales' or c == 'emp_admin' or c == 'emp_farm' or c == 'emp_construct' or c == 'emp_repair' or c == 'emp_production' or c == 'emp_transport' or c == 'emp_material':
            perc.append(c)
            sub_set.loc[:, c] = np.array(sub_set[c])/(emp_rate * pop)
            sub_set.rename(index=str, columns={c:c+'_%'}, inplace = True)
        if c == 'occ_lt0.5' or c == 'occ_0.5to1.0' or c == 'occ_1.01to1.5'	or c == 'occ_1.51to2.0'	or c == 'occ_gt2.01' or c == 'house_noplumbing'	or c == 'house_nokitchen' or c == 'house_income_gt30_USD':
            perc.append(c)
            sub_set.loc[:,c] = np.array(sub_set[c])/household_no
            sub_set.rename(index=str, columns={c:c+'_%'}, inplace = True)
'''
sub_set = pd.read_csv('data/health_determinant.csv', index_col=0)

columns = list(sub_set)

for c in columns:
    sub_set.loc[:,c] = (sub_set[c]-sub_set[c].mean())#/sub_set[c].std()

sub_set.to_csv('data/determinant_data_mn.csv')
'''
sub_set.set_index('Unnamed: 0', inplace = True)
sub_set.to_csv('../cleaned_sub_1_raw.csv')

print("COLUMNS DROPPED DUE TO INCORRECT DATA")
print(inc_data)
print('\n')

print("COLUMNS DROPPED DUE TO SPARSE DATA")
print(sparse)
print('\n')

print("COLUMNS THAT HAD TO BE CLEANED")
print(clean)
print('\n')

print("COLUMNS THAT HAD TO BE CONVERTED TO A RATE")
print(rate)
print('\n')

print("COLUMNS THAT HAD TO BE NORMALIZED")
print(norm)
print('\n')

print("COLUMNS THAT HAD TO BE CONVERTED TO PERCENT")
print(perc)
print('\n')
'''