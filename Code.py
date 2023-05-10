#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 4/28/2023
Author: Team 26

"""
import pandas as pd	
import numpy as np
from linearmodels.panel import PanelOLS
import wrds
from scipy.stats.mstats import winsorize
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score


###############################COMPANY DATA#################################
##################Extracting company data from WRDS##########################
conn = wrds.Connection()
db = 'compd'
table = 'funda'
conn.list_libraries().sort()
type(conn.list_libraries())
conn.list_tables(library='comp_na_daily_all')

company_data = conn.raw_sql("""select *
                            
                            from comp.fundq a 
                            inner join comp.company b 
                            
                            on a.gvkey = b.gvkey
                            and a.conm = b.conm
                        
                           
                            """)
                            
                            
########################Company Data Cleaning##################################                            
# Dropping rows that has no gsector
company_data = company_data.dropna(subset =['gsector'], how ='all')

#Summary for whole dataframe
company_subset = company_data[['fyearq','fqtr','tic','gind','gsector','gsubind','epspiq','prccq','niq','cogsq','cshiq','cshoq','piq','cimiiy','ciy','iby','opepsy','oiadpy','oibdpy','piy']].sort_values(['fyearq','fqtr','tic'])

company_subset_1 = pd.concat([company_subset,pd.get_dummies(company_subset['gsector'])],axis=1)

company_subset_1 = company_subset_1.rename(columns={10.0:'energy',15.0:'materials',20.0:'industrials',25.0:'consdisc',  30.0:'consstaples',35.0:'healthcare',40.0:'financials',45.0:'infotech',50.0:'communications',55.0:'utilities',60.0:'realestate'})

company_subset_1.drop(company_subset_1[company_subset_1['fyearq']<1965].index,inplace=True)

#Read final company data
company_data = pd.read_csv(r'C:\Users\ALIENWARE\Downloads\CompanyData2.csv')

company_data.columns.tolist()

#Drop NA values for tic
company_data = company_data[company_data['tic'].notna()]
company_data[["tic"]]

company_data.describe()

company_data.sort_values(by=["fyearq"])
company_subset_2 = company_data

#Cleaning company_subset_2 : to remove all NAs
company_subset_2_filtered = company_subset_2.groupby('tic').filter(lambda x : not x['epspiq'].isnull().all())

# define function to remove NA values from the beginning and end of a DataFrame
def remove_na(df):
    print(df['tic'])
    return df.loc[df['epspiq'].first_valid_index():df['epspiq'].last_valid_index()]

# group by 'group' column and apply 'remove_na' function to each group
CompanyData_filtered2 = company_subset_2_filtered.groupby('tic').apply(remove_na)

#Remove NA from fqtr
CompanyData_filtered2 = CompanyData_filtered2[CompanyData_filtered2['fqtr'].notna()]

CompanyData_filtered2['fyearq'] = CompanyData_filtered2['fyearq'].astype(int)
CompanyData_filtered2['fqtr']   = CompanyData_filtered2['fqtr'].astype(int)

# Combine 'year' and 'quarter' columns into a single string column
CompanyData_filtered2['date_str'] = CompanyData_filtered2['fyearq'].astype(str) + '-Q' + CompanyData_filtered2['fqtr'].astype(str)

# Convert date to datetime
CompanyData_filtered2['date'] = pd.to_datetime(CompanyData_filtered2['date_str'])


#############################Company Data Preparation##########################
# Define a function to interpolate missing values using a spline
def interpolate_spline(group):
    # Create a mask for missing values
    mask = group['epspiq'].isna()
    # Create a copy of the group to avoid modifying the original dataframe
    group_copy = group.copy()
    # Use spline interpolation to fill missing values
    group_copy.loc[mask, 'epspiq'] = np.interp(
        group_copy['date'][mask].astype(np.int64),
        group_copy['date'][~mask].astype(np.int64),
        group_copy['epspiq'][~mask])
    return group_copy

CompanyData_filtered2 = CompanyData_filtered2.reset_index(drop=True)

# Apply the interpolation function to each group
interpolated = CompanyData_filtered2.groupby('tic').apply(interpolate_spline)
save1 = interpolated
interpolated.head()

#Drop fyearq = 2023
interpolated = interpolated.drop(interpolated[interpolated['fyearq']==2023].index)

#Create Period column
interpolated['period'] = interpolated['fyearq'] + (interpolated['fqtr']-1)/4

interpolated = interpolated.reset_index(drop=True)

#Drop tic if cshoq is NA for all rows:
interpolated_filtered = interpolated.groupby('tic').filter(lambda x : not x['cshoq'].isnull().all())

#Interpolate CSHOQ
# Define a function to interpolate missing values using a spline
def interpolate_splineCSHOQ(group):
    # Create a mask for missing values
    mask = group['cshoq'].isna()
    
    # Create a copy of the group to avoid modifying the original dataframe
    group_copy = group.copy()
    # Use spline interpolation to fill missing values
    group_copy.loc[mask, 'cshoq'] = np.interp(
        group_copy['period'][mask],
        group_copy['period'][~mask],
        group_copy['cshoq'][~mask])
    return group_copy

interpolated_filtered.head()

# Apply the interpolation function to each group
interpolated_cshoq = interpolated_filtered.groupby('tic').apply(interpolate_splineCSHOQ)

#Create size dummy variables according to year and quater
interpolated_cshoq['earnings'] = interpolated_cshoq['epspiq']*interpolated_cshoq['cshoq']
interpolated_cshoq['group'] = interpolated_cshoq.groupby('period')['earnings'].transform(lambda x: pd.cut(x, bins=[-float('inf'), 0.33, 0.66, float('inf')], labels=['small', 'medium', 'large'], duplicates='drop'))

interpolated_cshoq['size_small'] = [1 if x =='small' else 0 for x in interpolated_cshoq['group']]
interpolated_cshoq['size_medium'] = [1 if x =='medium' else 0 for x in interpolated_cshoq['group']]
interpolated_cshoq['size_large'] = [1 if x =='large' else 0 for x in interpolated_cshoq['group']]

interpolated_cshoq = interpolated_cshoq.reset_index(drop=True)

#Transformation on EPS
# Winsorize the data to the 95th percentile
interpolated_cshoq['epspiq'] = winsorize(interpolated_cshoq['epspiq'], limits=(0.01, 0.01))

##Saving interpolated cshoq in a new df incase anything goes wrong
interpolated_cshoq_save = interpolated_cshoq
interpolated_cshoq1 = interpolated_cshoq

interpolated_cshoq1['date'] = pd.to_datetime(interpolated_cshoq1['date'])
interpolated_cshoq1.set_index('date', inplace=True)

#Dropping companies which have less than 8 observations for the tic 
group_company_data = interpolated_cshoq.groupby(['tic'])
filtered_data = interpolated_cshoq.groupby(['tic']).filter(lambda x : len(x) >= 8)
filtered_data1 = filtered_data.set_index(['tic', filtered_data.index])

print(filtered_data.head())
group_company_data1 = filtered_data

def seasonal_adjustment(df):
    decomposition = seasonal_decompose(df['epspiq'], model = 'additive', period=4)
    return decomposition.observed - decomposition.seasonal
  
adjusted_data = group_company_data1.groupby(['tic']).apply(seasonal_adjustment)
adjusted_data.name = 'eps_sa'
filtered_data1 = pd.merge(filtered_data1, adjusted_data, on=['tic', 'date'])
interpolated_cshoq = filtered_data1

eps_min = min(interpolated_cshoq['eps_sa'])
eps_max = max(interpolated_cshoq['eps_sa'])

interpolated_cshoq['epspiq_tf'] = ((interpolated_cshoq['eps_sa']-eps_min)/(eps_max-eps_min))*4+1

#Save company data
interpolated_cshoq['epspiq_lag'] = interpolated_cshoq.groupby('tic')['epspiq_tf'].shift(1)

clean_data = interpolated_cshoq




###################################CPI DATA####################################
##############################CPI Data Preparation#############################
#Import CPI Data
cpi_data = pd.read_csv(r'C:\Users\ALIENWARE\Downloads\cpi_data.csv')

# convert the date column to a datetime format
cpi_data['DATE'] = pd.to_datetime(cpi_data['DATE'])

# extract the year and quarter from the date column
cpi_data['fyearq'] = cpi_data['DATE'].dt.year
cpi_data['fqtr'] = cpi_data['DATE'].dt.quarter
cpi_data = cpi_data.rename(columns={'CPALTT01USQ661S': 'cpi'})
cpi_data.head()

#Seasonality
# Perform seasonal decomposition
cpi_data["DATE"]=pd.to_datetime(cpi_data["DATE"])
cpi_data.set_index("DATE",inplace=True)
decomp = sm.tsa.seasonal_decompose(cpi_data['cpi'], model="additive")

# Plot the components
decomp.plot()
decomp.seasonal
cpi_sa = decomp.observed-decomp.seasonal
cpi_data['cpi_sa'] = cpi_sa
print(cpi_sa)

# Check seasonality after removing it
decomp = sm.tsa.seasonal_decompose(cpi_data['cpi_sa'], model="additive")

# Plot the components
decomp.plot()
decomp.seasonal

#Create CPI percent change and difference
cpi_data['cpi_pct_change'] = cpi_data['cpi_sa'].pct_change()
cpi_data['cpi_pct_change'][0] = 0

cpi_data['cpi_diff'] = 0
cpi_data['cpi_diff'][1:] = np.diff(cpi_data['cpi_sa'])
cpi_data

#Running tests for CPI
# Plot the ACF and PACF
fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(cpi_data["cpi_sa"], ax=axes[0], title="ACF of CPI")
plot_pacf(cpi_data["cpi_sa"], ax=axes[1], title="PACF of CPI")
plt.tight_layout()

fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(cpi_data["cpi_pct_change"], ax=axes[0], title="ACF of CPI PCT Change")
plot_pacf(cpi_data["cpi_pct_change"], ax=axes[1], title="PACF of CPI PCT Change")
plt.tight_layout()

fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(cpi_data["cpi_diff"], ax=axes[0], title="ACF of CPI Diff")
plot_pacf(cpi_data["cpi_diff"], ax=axes[1], title="PACF of CPI Diff")
plt.tight_layout()

# Run the ADF test (p_value should be < 0.05), take the pct_change data
result = adfuller(cpi_data["cpi_sa"])

# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Run the ADF test
result = adfuller(cpi_data["cpi_pct_change"])

# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

result = adfuller(cpi_data["cpi_diff"])

# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

#To check trend (Stationarity)
cpi_data['period'] = cpi_data['fyearq'] + (cpi_data['fqtr']-1)/4

plt.plot(cpi_data["period"],cpi_data["cpi_pct_change"])
plt.show()

# Run the KPSS test (p_value should be > 0.05)
result = kpss(cpi_data["cpi_sa"], regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

result = kpss(cpi_data["cpi_pct_change"], regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

result = kpss(cpi_data["cpi_diff"], regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

#ARMA code for CPI
# define the range of p, d, and q values
p_values = range(0, 8)
d_values = range(0, 8)
q_values = range(0, 8)

# initialize variables to store the best model and its AIC value
best_model = None
best_aic = float('inf')

# iterate over different combinations of p, d, and q values
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # fit the ARIMA model to the time series data
                model = ARIMA(cpi_data['cpi_pct_change'], order=(p, d, q))
                model_fit = model.fit()

                # compare the AIC value of the current model with the best model so far
                aic = model_fit.aic
                if aic < best_aic:
                    best_model = model_fit
                    best_aic = aic
            except:
                continue

# print the summary of the best model
print(best_model.summary())

residuals = best_model.resid

plot_acf(residuals)
plot_pacf(residuals)

cpi_data['cpi_arima']=residuals

best_model.plot_diagnostics(figsize=(7,5))
plt.show()


###################################PPI DATA####################################
##############################PPI Data Preparation#############################
#Import PPI Data
ppi_data = pd.read_csv(r'C:\Users\ALIENWARE\Downloads\PITGCG01USQ661N.csv')
ppi_data['DATE'] = pd.to_datetime(ppi_data['DATE'])
# extract the year and quarter from the date column
ppi_data['fyearq'] = ppi_data['DATE'].dt.year
ppi_data['fqtr'] = ppi_data['DATE'].dt.quarter
ppi_data = ppi_data.rename(columns={'PITGCG01USQ661N': 'ppi'})
ppi_data.head()

#Seasonality
# Perform seasonal decomposition
ppi_data["DATE"]=pd.to_datetime(ppi_data["DATE"])
ppi_data.set_index("DATE",inplace=True)
decomp = sm.tsa.seasonal_decompose(ppi_data['ppi'], model="additive")

# Plot the components
decomp.plot()
decomp.seasonal
ppi_sa = decomp.observed-decomp.seasonal
print(ppi_sa)

ppi_data['ppi_sa'] = ppi_sa

# Check seasonality after removing it
decomp = sm.tsa.seasonal_decompose(ppi_data['ppi_sa'], model="additive")

# Plot the components
decomp.plot()
decomp.seasonal

#Percent Change and diff for PPI
ppi_data['ppi_pct_change'] = ppi_data['ppi_sa'].pct_change()
ppi_data['ppi_pct_change'][0] = 0

ppi_data['ppi_diff'] = 0
ppi_data['ppi_diff'][1:] = np.diff(ppi_data['ppi_sa'])
ppi_data

#Running tests for PPI
# Plot the ACF and PACF
fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(ppi_data["ppi_sa"], ax=axes[0], title="ACF of PPI")
plot_pacf(ppi_data["ppi_sa"], ax=axes[1], title="PACF of PPI")
plt.tight_layout()

fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(ppi_data["ppi_pct_change"], ax=axes[0], title="ACF of PPI pct Change")
plot_pacf(ppi_data["ppi_pct_change"], ax=axes[1], title="PACF of PPI pct Change")
plt.tight_layout()

fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(ppi_data["ppi_diff"], ax=axes[0], title="ACF of PPI Diff")
plot_pacf(ppi_data["ppi_diff"], ax=axes[1], title="PACF of PPI Diff")
plt.tight_layout()

# Run the ADF test (p_value should be < 0.05), take the pct_change data
result = adfuller(ppi_data["ppi_sa"])

# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Run the ADF test
result = adfuller(ppi_data["ppi_pct_change"])

# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

result = adfuller(ppi_data["ppi_diff"])

# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

#To check trend (Stationarity)
ppi_data['period'] = ppi_data['fyearq'] + (ppi_data['fqtr']-1)/4
plt.plot(ppi_data['period'],ppi_data["ppi_pct_change"])
plt.show()

# Run the KPSS test (p_value should be > 0.05)
result = kpss(ppi_data["ppi_sa"], regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

result = kpss(ppi_data["ppi_pct_change"], regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

result = kpss(ppi_data["ppi_diff"], regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

#ARMA code for PPI
# define the range of p, d, and q values
p_values = range(0, 8)
d_values = range(0, 8)
q_values = range(0, 8)

# initialize variables to store the best model and its AIC value
best_model = None
best_aic = float('inf')

# iterate over different combinations of p, d, and q values
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # fit the ARIMA model to the time series data
                model = ARIMA(ppi_data['ppi_pct_change'], order=(p, d, q))
                model_fit = model.fit()

                # compare the AIC value of the current model with the best model so far
                aic = model_fit.aic
                if aic < best_aic:
                    best_model = model_fit
                    best_aic = aic
            except:
                continue

# print the summary of the best model
print(best_model.summary())

residuals = best_model.resid

plot_acf(residuals)
plot_pacf(residuals)

ppi_data['ppi_arima']=residuals

best_model.plot_diagnostics(figsize=(7,5))
plt.show()


##########################Create Unbalanced Panel Model########################
clean_data = clean_data.reset_index()
merged_data = pd.merge(clean_data, cpi_data, how='left', on=['fyearq', 'fqtr'])
merged_data = pd.merge(merged_data, ppi_data, how='left', on=['fyearq', 'fqtr'])

merged_data['returns_eps'] =  (merged_data['epspiq_tf']/merged_data['epspiq_lag']) -1
merged_data['returns_lag'] = merged_data.groupby('tic')['returns_eps'].shift(1)
merged_data = merged_data.dropna(subset=['epspiq_lag', 'returns_eps','returns_lag'])
merged_data.to_csv("mergeddata.csv")

merged = pd.read_csv(r'mergeddata.csv')
merged = merged_data

merged.set_index(['tic','date'],inplace=True)
merged.index.set_levels(pd.to_datetime(merged.index.levels[1]), level=1, inplace=True)

#Seasonality check and Stationarity checks for EPS
clean_data_EPS = clean_data.groupby('date')['epspiq'].mean()
fig,axs = plt.subplots()
axs.plot(clean_data_EPS, color='blue')
axs.set_xlabel('Date')
plt.show()

clean_data_EPS = merged_data.groupby('date')['eps_sa'].mean()
fig,axs = plt.subplots()
axs.plot(clean_data_EPS, color='blue')
axs.set_xlabel('Date')
plt.show()

result = adfuller(merged_data.groupby('date')['eps_sa'].mean())
# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

result = adfuller(merged_data.groupby('date')['returns_eps'].mean())
# Print the test statistic and p-value
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Run the KPSS test (p_value should be > 0.05)
result = kpss(merged_data.groupby('date')['eps_sa'].mean(), regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])

# Run the KPSS test (p_value should be > 0.05)
result = kpss(merged_data.groupby('date')['returns_eps'].mean(), regression="c", nlags="auto")

# Print the test statistic and p-value
print("KPSS Statistic:", result[0])
print("p-value:", result[1])




###############################EDA#############################################
###############Plot EPS Mean vs. CPI and PPI###################################
merged_data_small = merged_data[merged_data['size_small']==1]
merged_data_medium = merged_data[merged_data['size_medium']==1]
merged_data_large = merged_data[merged_data['size_large']==1]

merged_data_small_mean = merged_data_small.groupby('date')['returns_eps'].mean()
merged_data_medium_mean = merged_data_medium.groupby('date')['returns_eps'].mean()
merged_data_large_mean = merged_data_large.groupby('date')['returns_eps'].mean()

##CPI plots
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# plot EPS small mean on the first y-axis
axs[0].plot(merged_data_small_mean, color='blue')
axs[0].set_title('Overall Small')
axs[0].set_ylabel('EPS_pct_change_small', color='blue')
ax2 = axs[0].twinx()
ax2.plot(cpi_data['cpi_pct_change'], color='red')
ax2.set_ylabel('CPI', color='red')
axs[0].set_xlabel('Date')

# plot EPS medium mean on the first y-axis
axs[1].plot(merged_data_medium_mean, color='blue')
axs[1].set_title('Overall Medium')
axs[1].set_ylabel('EPS_pct_change_medium', color='blue')
ax2 = axs[1].twinx()
ax2.plot(cpi_data['cpi_pct_change'], color='red')
ax2.set_ylabel('CPI', color='red')
axs[1].set_xlabel('Date')

# plot EPS large mean on the first y-axis
axs[2].plot(merged_data_large_mean, color='blue')
axs[2].set_title('Overall Large')
axs[2].set_ylabel('EPS_pct_change_large', color='blue')
ax2 = axs[2].twinx()
ax2.plot(cpi_data['cpi_pct_change'], color='red')
ax2.set_ylabel('CPI', color='red')
axs[2].set_xlabel('Date')
plt.show()

##PPI plots
fig, axs = plt.subplots(3, 1, figsize=(8, 10))

# plot EPS small mean on the first y-axis
axs[0].plot(merged_data_small_mean, color='blue')
axs[0].set_title('Overall Small')
axs[0].set_ylabel('EPS_pct_change_small', color='blue')
ax2 = axs[0].twinx()
ax2.plot(ppi_data['ppi_pct_change'], color='orange')
ax2.set_ylabel('PPI', color='orange')
axs[0].set_xlabel('Date')

# plot EPS medium mean on the first y-axis
axs[1].plot(merged_data_medium_mean, color='blue')
axs[1].set_title('Overall Medium')
axs[1].set_ylabel('EPS_pct_change_medium', color='blue')
ax2 = axs[1].twinx()
ax2.plot(ppi_data['ppi_pct_change'], color='orange')
ax2.set_ylabel('PPI', color='orange')
axs[1].set_xlabel('Date')

# plot EPS large mean on the first y-axis
axs[2].plot(merged_data_large_mean, color='blue')
axs[2].set_title('Overall Large')
axs[2].set_ylabel('EPS_pct_change_large', color='blue')
ax2 = axs[2].twinx()
ax2.plot(ppi_data['ppi_pct_change'], color='orange')
ax2.set_ylabel('PPI', color='orange')
axs[2].set_xlabel('Date')
plt.show()


#################Correlation of EPS with CPI and PPI###########################
corr_df = merged_data
merged['cpi_pct_change'].corr(merged['returns_eps'])
merged['ppi_pct_change'].corr(merged['returns_eps'])

clean_data_small = corr_df[corr_df['size_small']==1]
clean_data_medium = corr_df[corr_df['size_medium']==1]
clean_data_large = corr_df[corr_df['size_large']==1]

clean_data_small_mean = clean_data_small.groupby('period')['returns_eps'].mean()
clean_data_medium_mean = clean_data_medium.groupby('period')['returns_eps'].mean()
clean_data_large_mean =clean_data_large.groupby('period')['returns_eps'].mean()

def calculate_correlation(corr_df, industry_type, industry_size):
    data = corr_df[corr_df[industry_type] == 1]
    
    if industry_size == 'small':
        data_size = data
    elif industry_size == 'medium':
        data_size = data
    elif industry_size == 'large':
        data_size = data
    else:
        print('Invalid energy size group')
        return None
    
    data_size_mean = data_size.groupby('date')['returns_eps'].mean()
    data_size_mean= data_size_mean.reset_index()
    data_size_mean = pd.DataFrame(data_size_mean)
    
    cccor = pd.merge(data_size_mean, cpi_data, left_on='date', right_on = 'DATE')
    cccor.columns.tolist()
    corr_coef = np.corrcoef(cccor['cpi_pct_change'], cccor['returns_eps'])[0, 1]
    
    return corr_coef

corr_df.columns.to_list()

print("Energy")
print(calculate_correlation(corr_df,'energy','small'))
print(calculate_correlation(corr_df,'energy','medium'))
print(calculate_correlation(corr_df,'energy','large'))

print("Materials")
print(calculate_correlation(corr_df,'materials','small'))
print(calculate_correlation(corr_df,'materials','medium'))
print(calculate_correlation(corr_df,'materials','large'))

print("Industrials")
print(calculate_correlation(corr_df,'industrials','small'))
print(calculate_correlation(corr_df,'industrials','medium'))
print(calculate_correlation(corr_df,'industrials','large'))

print("Consdisc")
print(calculate_correlation(corr_df,'consdisc','small'))
print(calculate_correlation(corr_df,'consdisc','medium'))
print(calculate_correlation(corr_df,'consdisc','large'))

print("Consstaples")
print(calculate_correlation(corr_df,'consstaples','small'))
print(calculate_correlation(corr_df,'consstaples','medium'))
print(calculate_correlation(corr_df,'consstaples','large'))

print("Healthcare")
print(calculate_correlation(corr_df,'healthcare','small'))
print(calculate_correlation(corr_df,'healthcare','medium'))
print(calculate_correlation(corr_df,'healthcare','large'))

print("Financials")
print(calculate_correlation(corr_df,'financials','small'))
print(calculate_correlation(corr_df,'financials','medium'))
print(calculate_correlation(corr_df,'financials','large'))

print("Infotech")
print(calculate_correlation(corr_df,'infotech','small'))
print(calculate_correlation(corr_df,'infotech','medium'))
print(calculate_correlation(corr_df,'infotech','large'))

print("Communications")
print(calculate_correlation(corr_df,'communications','small'))
print(calculate_correlation(corr_df,'communications','medium'))
print(calculate_correlation(corr_df,'communications','large'))

print("Utilities")
print(calculate_correlation(corr_df,'utilities','small'))
print(calculate_correlation(corr_df,'utilities','medium'))
print(calculate_correlation(corr_df,'utilities','large'))

print("Realestate")
print(calculate_correlation(corr_df,'realestate','small'))
print(calculate_correlation(corr_df,'realestate','medium'))
print(calculate_correlation(corr_df,'realestate','large'))




################################Modeling#######################################
###########################Model Construction##################################
#Remove financials and size-mid to avoid dummy trap
def calculate_aic_bic(results):
    log_likelihood = results.loglik
    num_parameters = len(results.params)
    n = results.nobs

    aic = -2 * log_likelihood + 2 * num_parameters
    bic = -2 * log_likelihood + num_parameters * np.log(n)

    print("AIC:", aic)
    print("BIC:", bic)

#Baseline Model
formula0 = 'returns_eps ~ 1 + cpi_pct_change + ppi_pct_change'
model0 = PanelOLS.from_formula(formula0, data=merged, drop_absorbed=True)
results0 = model0.fit()

# print regression results
print(results0)
calculate_aic_bic(results0)

#Model 1
formula1 ='returns_eps ~ 1 + returns_lag + cpi_pct_change + ppi_pct_change'
model1 = PanelOLS.from_formula(formula1, data=merged, drop_absorbed=True)
results1 = model1.fit()

# print regression results
print(results1)
calculate_aic_bic(results1)

#Model 2
formula2 = 'returns_eps ~ 1+returns_lag + cpi_pct_change+ ppi_pct_change+\
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate'
 
model2 = PanelOLS.from_formula(formula2, data=merged, drop_absorbed=True)
results2 = model2.fit()

# print regression results
print(results2)
calculate_aic_bic(results2)

#Model 3
formula3 = 'returns_eps ~ 1+returns_lag + cpi_pct_change+ ppi_pct_change+ size_small + size_large'
 
model3 = PanelOLS.from_formula(formula3, data=merged, drop_absorbed=True)
results3 = model3.fit()

# print regression results
print(results3)
calculate_aic_bic(results3)

#Model 4
formula4 = 'returns_eps ~ 1+returns_lag + cpi_pct_change+ ppi_pct_change+\
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate+\
            ppi_pct_change:energy + ppi_pct_change:materials +\
            ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
            ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
            ppi_pct_change:utilities + ppi_pct_change:realestate' 

model4 = PanelOLS.from_formula(formula4, data=merged, drop_absorbed=True)
results4 = model4.fit()

# print regression results
print(results4)
calculate_aic_bic(results4)

#Model 4_CPI
formula4_CPI = 'returns_eps ~ 1+returns_lag + cpi_pct_change+ \
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate' 

model4_CPI = PanelOLS.from_formula(formula4_CPI, data=merged, drop_absorbed=True)
results4_CPI = model4_CPI.fit()

# print regression results
print(results4_CPI)
calculate_aic_bic(results4_CPI)

#Model 4_PPI
formula4_PPI = 'returns_eps ~ 1+returns_lag + ppi_pct_change+\
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
            ppi_pct_change:energy + ppi_pct_change:materials +\
            ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
            ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
            ppi_pct_change:utilities + ppi_pct_change:realestate' 

model4_PPI = PanelOLS.from_formula(formula4_PPI, data=merged, drop_absorbed=True)
results4_PPI = model4_PPI.fit()

# print regression results
print(results4_PPI)
calculate_aic_bic(results4_PPI)

#Model 4_Small
formula4_S = 'returns_eps ~ 1+ returns_lag + cpi_pct_change+ ppi_pct_change+\
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate +\
     ppi_pct_change:energy + ppi_pct_change:materials +\
     ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
     ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
     ppi_pct_change:utilities + ppi_pct_change:realestate'

model4_S = PanelOLS.from_formula(formula4_S, data=merged[merged['size_small']==1], drop_absorbed=True)
results4_S = model4_S.fit()

# print regression results
print(results4_S)
calculate_aic_bic(results4_S)

#Model 4_Medium
formula4_M = 'returns_eps ~ 1+ returns_lag + cpi_pct_change+ ppi_pct_change+\
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate +\
     ppi_pct_change:energy + ppi_pct_change:materials +\
     ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
     ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
     ppi_pct_change:utilities + ppi_pct_change:realestate'

model4_M = PanelOLS.from_formula(formula4_M, data=merged[merged['size_medium']==1], drop_absorbed=True)
results4_M = model4_M.fit()

# print regression results
print(results4_M)
calculate_aic_bic(results4_M)

#Model 4_Large
formula4_L = 'returns_eps ~ 1+ returns_lag + cpi_pct_change+ ppi_pct_change+\
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate +\
     ppi_pct_change:energy + ppi_pct_change:materials +\
     ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
     ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
     ppi_pct_change:utilities + ppi_pct_change:realestate'

model4_L = PanelOLS.from_formula(formula4_L, data=merged[merged['size_large']==1], drop_absorbed=True)
results4_L = model4_L.fit()

# print regression results
print(results4_L)
calculate_aic_bic(results4_L)

#Model 5
formula5 = 'returns_eps ~ 1+returns_lag + ppi_pct_change+ cpi_pct_change+ size_small + size_large +\
   cpi_pct_change:size_small+cpi_pct_change:size_large + \
       ppi_pct_change:size_small+ppi_pct_change:size_large'

model5 = PanelOLS.from_formula(formula5, data=merged, drop_absorbed=True)
results5 = model5.fit()

# print regression results
print(results5)
calculate_aic_bic(results5)

#Model 6
formula6 = 'returns_eps ~ 1+ returns_lag + ppi_pct_change+ cpi_pct_change+ \
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate +\
    size_small+size_large+\
        cpi_pct_change:size_small+cpi_pct_change:size_large+\
            ppi_pct_change:energy + ppi_pct_change:materials +\
            ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
            ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
            ppi_pct_change:utilities + ppi_pct_change:realestate +\
                   ppi_pct_change:size_small+ppi_pct_change:size_large'

model6 = PanelOLS.from_formula(formula6, data=merged, drop_absorbed=True)
results6 = model6.fit()

# print regression results
print(results6)
calculate_aic_bic(results6)

#Model 6_CPI
formula6_CPI = 'returns_eps ~ 1+ returns_lag + cpi_pct_change+ \
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
 cpi_pct_change:energy + cpi_pct_change:materials +\
 cpi_pct_change:industrials + cpi_pct_change:consdisc + cpi_pct_change:consstaples+ \
 cpi_pct_change:healthcare + cpi_pct_change:infotech + cpi_pct_change:communications + \
 cpi_pct_change:utilities + cpi_pct_change:realestate +\
    size_small+size_large+\
        cpi_pct_change:size_small+cpi_pct_change:size_large'

model6_CPI = PanelOLS.from_formula(formula6_CPI, data=merged, drop_absorbed=True)
results6_CPI = model6_CPI.fit()

# print regression results
print(results6_CPI)
calculate_aic_bic(results6_CPI)

#Model 6_PPI
formula6_PPI = 'returns_eps ~ 1+ returns_lag + ppi_pct_change+ \
 energy + materials + industrials + consdisc + consstaples+ healthcare + infotech + communications + utilities + realestate+\
            ppi_pct_change:energy + ppi_pct_change:materials +\
            ppi_pct_change:industrials + ppi_pct_change:consdisc + ppi_pct_change:consstaples+ \
            ppi_pct_change:healthcare + ppi_pct_change:infotech + ppi_pct_change:communications + \
            ppi_pct_change:utilities + ppi_pct_change:realestate +\
                   ppi_pct_change:size_small+ppi_pct_change:size_large'

model6_PPI = PanelOLS.from_formula(formula6_PPI, data=merged, drop_absorbed=True)
results6_PPI = model6_PPI.fit()

# print regression results
print(results6_PPI)
calculate_aic_bic(results6_PPI)


#############################Model Evaluation##################################
#Residual Analysis
residuals=results6.resids

# Load data and calculate autocorrelation
acf = sm.tsa.stattools.acf(residuals, nlags=60)

# Calculate confidence intervals
conf_int = 1.96 / len(residuals)**0.5

# Plot ACF with confidence intervals
fig, ax = plt.subplots(figsize=(8,3))
sm.graphics.tsa.plot_acf(residuals, ax=ax, lags=60)
ax.axhline(y=conf_int, linestyle='--', color='gray')
ax.axhline(y=-conf_int, linestyle='--', color='gray')
plt.show()

# Plot PACF with confidence intervals
fig, ax = plt.subplots(figsize=(8,3))
sm.graphics.tsa.plot_pacf(residuals, ax=ax, lags=60)
ax.axhline(y=conf_int, linestyle='--', color='gray')
ax.axhline(y=-conf_int, linestyle='--', color='gray')
plt.show()

# Fit the model0 and obtain the residuals
residuals6 = results6.resids

# plot a scatter plot of the residuals
fig, ax = plt.subplots()
ax.scatter(range(len(residuals6)), residuals6, s=10)
ax.set_xlabel('Observation')
ax.set_ylabel('Residual')
ax.set_title('Residual scatter plot')
plt.show()

fig, axes = plt.subplots(nrows=2, figsize=(8, 6))
plot_acf(residuals6, ax=axes[0], title="ACF of CPI PCT Change")
plot_pacf(residuals6, ax=axes[1], title="PACF of CPI PCT Change")
plt.tight_layout()

#ARIMA Model
p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)

# initialize variables to store the best model and its AIC value
best_model = None
best_aic = float('inf')

# iterate over different combinations of p, d, and q values
for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                # fit the ARIMA model to the time series data
                model = ARIMA(residuals, order=(p, d, q))
                model_fit = model.fit()

                # compare the AIC value of the current model with the best model so far
                aic = model_fit.aic
                if aic < best_aic:
                    best_model = model_fit
                    best_aic = aic
            except:
                continue

# print the summary of the best model
print(best_model.summary())

residuals = residuals6

plot_acf(residuals)
plot_pacf(residuals)


####################################Cross Validation###########################
merged['cpi_pct_change:energy'] = merged['cpi_pct_change'] * merged['energy']
merged['cpi_pct_change:materials'] = merged['cpi_pct_change'] * merged['materials']
merged['cpi_pct_change:industrials'] = merged['cpi_pct_change'] * merged['industrials']
merged['cpi_pct_change:consdisc'] = merged['cpi_pct_change'] * merged['consdisc']
merged['cpi_pct_change:consstaples'] = merged['cpi_pct_change'] * merged['consstaples']
merged['cpi_pct_change:healthcare'] = merged['cpi_pct_change'] * merged['healthcare']
merged['cpi_pct_change:infotech'] = merged['cpi_pct_change'] * merged['infotech']
merged['cpi_pct_change:communications'] = merged['cpi_pct_change'] * merged['communications']
merged['cpi_pct_change:utilities'] = merged['cpi_pct_change'] * merged['utilities']
merged['cpi_pct_change:realestate'] = merged['cpi_pct_change'] * merged['realestate']
merged['cpi_pct_change:size_small'] = merged['cpi_pct_change'] * merged['size_small']
merged['cpi_pct_change:size_large'] = merged['cpi_pct_change'] * merged['size_large']
merged['ppi_pct_change:energy'] = merged['ppi_pct_change'] * merged['energy']
merged['ppi_pct_change:materials'] = merged['ppi_pct_change'] * merged['materials']
merged['ppi_pct_change:industrials'] = merged['ppi_pct_change'] * merged['industrials']
merged['ppi_pct_change:consdisc'] = merged['ppi_pct_change'] * merged['consdisc']
merged['ppi_pct_change:consstaples'] = merged['ppi_pct_change'] * merged['consstaples']
merged['ppi_pct_change:healthcare'] = merged['ppi_pct_change'] * merged['healthcare']
merged['ppi_pct_change:infotech'] = merged['ppi_pct_change'] * merged['infotech']
merged['ppi_pct_change:communications'] = merged['ppi_pct_change'] * merged['communications']
merged['ppi_pct_change:utilities'] = merged['ppi_pct_change'] * merged['utilities']
merged['ppi_pct_change:realestate'] = merged['ppi_pct_change'] * merged['realestate']
merged['ppi_pct_change:size_small'] = merged['ppi_pct_change'] * merged['size_small']
merged['ppi_pct_change:size_large'] = merged['ppi_pct_change'] * merged['size_large']

formula0_variables = [ 'cpi_pct_change', 'ppi_pct_change']
formula1_variables = [ 'returns_lag', 'cpi_pct_change', 'ppi_pct_change']
formula2_variables =[ 'returns_lag', 'cpi_pct_change', 'ppi_pct_change', 'energy', 'materials', 'industrials', 'consdisc', 'consstaples', 'healthcare', 'infotech', 'communications', 'utilities', 'realestate']
formula3_variables =[ 'returns_lag', 'cpi_pct_change', 'ppi_pct_change', 'size_small', 'size_large']
formula4_variables =[ 'returns_lag', 'cpi_pct_change', 'ppi_pct_change', 'energy', 'materials', 'industrials', 'consdisc', 'consstaples', 'healthcare', 'infotech', 'communications', 'utilities', 'realestate', 'cpi_pct_change:energy', 'cpi_pct_change:materials', 'cpi_pct_change:industrials', 'cpi_pct_change:consdisc', 'cpi_pct_change:consstaples', 'cpi_pct_change:healthcare', 'cpi_pct_change:infotech', 'cpi_pct_change:communications', 'cpi_pct_change:utilities', 'cpi_pct_change:realestate', 'ppi_pct_change:energy', 'ppi_pct_change:materials', 'ppi_pct_change:industrials', 'ppi_pct_change:consdisc', 'ppi_pct_change:consstaples', 'ppi_pct_change:healthcare', 'ppi_pct_change:infotech', 'ppi_pct_change:communications', 'ppi_pct_change:utilities', 'ppi_pct_change:realestate']
formula5_variables =[ 'returns_lag', 'ppi_pct_change', 'cpi_pct_change', 'size_small', 'size_large', 'cpi_pct_change:size_small', 'cpi_pct_change:size_large', 'ppi_pct_change:size_small', 'ppi_pct_change:size_large']
formula6_variables = [     'returns_lag',    'ppi_pct_change',    'cpi_pct_change',    'energy',    'materials',    'industrials',    'consdisc',    'consstaples',    'healthcare',    'infotech',    'communications',    'utilities',    'realestate',    'cpi_pct_change:energy',    'cpi_pct_change:materials',    'cpi_pct_change:industrials',    'cpi_pct_change:consdisc',    'cpi_pct_change:consstaples',    'cpi_pct_change:healthcare',    'cpi_pct_change:infotech',    'cpi_pct_change:communications',    'cpi_pct_change:utilities',    'cpi_pct_change:realestate',    'size_small',    'size_large',    'cpi_pct_change:size_small',    'cpi_pct_change:size_large',    'ppi_pct_change:energy',    'ppi_pct_change:materials',    'ppi_pct_change:industrials',    'ppi_pct_change:consdisc',    'ppi_pct_change:consstaples',    'ppi_pct_change:healthcare',    'ppi_pct_change:infotech',    'ppi_pct_change:communications',    'ppi_pct_change:utilities',    'ppi_pct_change:realestate',    'ppi_pct_change:size_small',    'ppi_pct_change:size_large']

formula_variables = [ formula0_variables, formula1_variables, formula2_variables,\
                      formula3_variables, formula4_variables, formula5_variables,\
                          formula6_variables]

formulas = [formula0, formula1, formula2, formula3, formula4, formula5, formula6]
formula_variables = [formula0_variables, formula1_variables, formula2_variables, formula3_variables, formula4_variables, formula5_variables, formula6_variables]
n_splits = 5

gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
groups = merged.index.get_level_values('tic')
test_r2_scores = []
test_aic_scores = []
test_bic_scores = []

tmp = [0,1,2,3,4,5,6]

for formula, formula_variables, i in zip(formulas, formula_variables, tmp):
    for train_index, test_index in gss.split(merged, merged['returns_eps'], groups):
        train_data = merged.iloc[train_index]
        test_data = merged.iloc[test_index]

        y_train = train_data['returns_eps']
        exog_train = train_data[formula_variables]
        exog_train = sm.add_constant(exog_train, prepend=True)
        exog_train = exog_train.reset_index().set_index(['tic', 'date'])

        y_test = test_data['returns_eps']
        exog_test = test_data[formula_variables]
        exog_test = sm.add_constant(exog_test, prepend=True)
        exog_test = exog_test.reset_index().set_index(['tic', 'date'])

        model = PanelOLS.from_formula(formula, data=train_data, drop_absorbed=True)
        results = model.fit()

        y_pred = results.predict(exog_test)

        test_r2 = r2_score(y_test, y_pred)
        test_r2_scores.append(test_r2)

        # Calculate AIC and BIC for the test set
        n = len(y_test)
        k = len(results.params)
        y_test = y_test.to_frame()
        y_test.columns = ['predictions']
        loglik = -n / 2 * (1 + np.log(2 * np.pi) + np.log((y_test - y_pred).T @ (y_test - y_pred) / n))
        loglik = loglik['predictions'].iloc[0]
        test_aic = 2 * k - 2 * loglik
        test_bic = k * np.log(n) - 2 * loglik
        test_aic_scores.append(test_aic)
        test_bic_scores.append(test_bic)

    print(f"Model: {i}")
    print(f"Mean test R^2 score: {pd.Series(test_r2_scores).mean()}")
    print(f"Mean test AIC score: {pd.Series(test_aic_scores).mean()}")
    print(f"Mean test BIC score: {pd.Series(test_bic_scores).mean()}")
    print()