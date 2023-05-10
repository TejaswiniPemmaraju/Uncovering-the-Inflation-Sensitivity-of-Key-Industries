# Uncovering the Inflation Sensitivity of Key Industries
An Analysis of PPI and CPI Effects on Adjusted EPS Using Unbalanced Panel Data Models. This was done under duke capstone project under the guidance of Kevin Walenta, Portfolio Manager, Fidelity Investments.

## Overview
The project, "Analyzing the Impact of Inflation Factors on Various Industries" was aimed to examine which United States industries are influenced by inflation factors and to identify the causes of the correlation between inflation and earnings. By understanding these relationships, businesses, investors, and policymakers could make informed decisions and develop effective strategies to navigate the complex economic landscape. The inflation factors utilized in the analysis were the Customer Pricing Index (CPI) and Producer Pricing Index (PPI). The earnings data used in the analysis was represented by the earnings per share (EPS) of each company over a period spanning from 1964 to 2022. 

## Steps Used
1. Data Selection
2. Data Preparation
3. Exploratory Data Analysis
4. Model selection and Construction
5. Model Evaluation

## Why Adjusted EPS over other performance metrics
Adjusted EPS is a more comprehensive measure of a company’s overall profitability while other measures are more focused on the company’s operating performance. 

## Data Cleaning
CPI and PPI data:
1. Checked missing values or significant outliers
Company Data:
1. Removed NAs for the non-operating period for all companies
2. Removed companies with all NAs for EPS and Cshoq
3. Removed short-period companies with less than 8 observations

## Data Preparation
CPI and PPI data & EPS data:
1. Decompose the data seasonally to remove seasonality. 
2. Used ADF, KPSS, ACF, and PACF tests to verify stationarity. Took percent change for CPI, PPI, and EPS for further analysis.
3. Interpolated EPS and CSHOQ to remove missing data and NAs.
4. Used Quantiles to classify the data into 3 different company sizes (Market Valuation: EPS* CSHOQ): Small, Mid, Large
5. Created dummy variables to indicate different industries.
6. Used Min Max method to transform EPS (Avoiding 0s before taking percent change)

## Model Selection
Used linear model for early analysis. Compared the balanced panel model, first difference model, fixed effects model, and unbalanced panel model. The unbalanced panel model fits the analysis better. Basic linear regression models were used to provide the first insights of the data, while the models listed later are all unbalanced panel models.

## Model Evaluation
To select the models, Performed linear regression and panel regression and calculated the R-squared of each models. Although model 6 is our best model, compared all the results from Model 1 to 6 to see if there is any commonality in the results.

## Findings
1. Individually PPI and CPI had an inverse relationship with EPS.
2. Movement of Finance EPS exhibited the most negative correlation with percentage change in CPI and PPI
3. Movement of Energy EPS exhibited the most positive correlation with percentage change in CPI and PPI
4. PPI had a direct relationship with EPS while CPI had an Inverse relationship with EPS for two factor models
5. The analysis showed Energy EPS performs well when there was a positive change in PPI, which was consistent in the earlier one factor and current two factor model.




