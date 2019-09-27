#!/usr/bin/env python
# coding: utf-8

#  Exploration HARP Performance Dataset

# The main goal of the code below is to get a better understanding of the raw HARP Performance Dataset

# In[2]:


# Import statements

import os
import datetime

import seaborn as sns
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split


# In[3]:


# See current working directory
cwd = os.getcwd()
cwd


# In[4]:


"""
Author: Nicolas Bulté
Solely the file path should be changed
"""
"""
Glossary mapping
"""

# LoanID               = Loan Identifier (A,P)
# MonthRep             = Monthly Reporting Period (P)
# Servicer             = Servicer Name (P)
# CurrInterestRate     = CURRENT INTEREST RATE (P)
# CAUPB                = CURRENT ACTUAL UNPAID PRINCIPAL BALANCE (P)
# LoanAge              = Loan Age (P)
# MonthsToMaturity     = Remaining Months to Legal Maturity (P)
# AdMonthsToMaturity   = ADJUSTED REMAINING MONTHS TO MATURITY (P)
# MaturityDate         = Maturity Date (P)
# MSA                  = Metropolitan Statistical Area (P)
# CLDS                 = Current Loan Delinquency Status (P)
# ModFlag              = Modification Flag (P)
# ZeroBalCode          = Zero Balance Code (P)
# ZeroBalDate          = Zero Balance Effective Date(P)
# LastInstallDate      = LAST PAID INSTALLMENT DATE
# ForeclosureDate      = FORECLOSURE DATE
# DispositionDate      = DISPOSITION DATE
# ForeclosureCosts     = FORECLOSURE COSTS (P)
# PPRC                 = Property Preservation and Repair Costs (P)
# AssetRecCost         = ASSET RECOVERY COSTS (P)
# MHEC                 = Miscellaneous Holding Expenses and Credits (P)
# ATFHP                = Associated Taxes for Holding Property (P)
# NetSaleProceeds      = Net Sale Proceeds (P)
# CreditEnhProceeds    = Credit Enhancement Proceeds (P)
# RPMWP                = Repurchase Make Whole Proceeds(P)
# OFP                  = Other Foreclosure Proceeds (P)
# NIBUPB               = Non-Interest Bearing UPB (P)
# PFUPB                = PRINCIPAL FORGIVENESS UPB (P)
# RMWPF                = Repurchase Make Whole Proceeds Flag (P)
# FPWA                 = Foreclosure Principal Write-off Amount (P)
# ServicingIndicator   = SERVICING ACTIVITY INDICATOR (P)


# Import datasets, select features and define the default-flag collumn.
col_per = ['LoanID', 'MonthRep', 'Servicer', 'CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity',
           'AdMonthsToMaturity', 'MaturityDate', 'MSA', 'CLDS', 'ModFlag', 'ZeroBalCode', 'ZeroBalDate',
           'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'ForeclosureCosts', 'PPRC', 'AssetRecCost', 'MHEC',
           'ATFHP', 'NetSaleProceeds', 'CreditEnhProceeds', 'RPMWP', 'OFP', 'NIBUPB', 'PFUPB', 'RMWPF',
           'FPWA', 'ServicingIndicator']

# Python will guess the datatypes not specified in the map function, for dates the dtype will be 'object'. (hence: here all dates)
# If an expected integer variables contains NaN values it will be set to 'float32'
perf_type_map = {'LoanID': 'int64', 'Servicer': 'category', 'CurrInterestRate': 'float32', 'CAUPB': 'float32',
                 'LoanAge': 'int64', 'MonthsToMaturity': 'int64', 'AdMonthsToMaturity': 'float32', 'MSA': 'category',
                 'CLDS': 'category', 'ModFlag': 'category', 'ZeroBalCode': 'float32', 'ForeclosureCosts': 'float32',
                 'PPRC': 'float32', 'AssetRecCost': 'float32', 'MHEC': 'float32', 'ATFHP': 'float32',
                 'NetSaleProceeds': 'float32', 'CreditEnhProceeds': 'float32', 'RPMWP': 'float32', 'OFP': 'float32',
                 'NIBUPB': 'float32', 'PFUPB': 'float32', 'RMWPF': 'category', 'FPWA': 'float32',
                 'ServicingIndicator': 'category'}

extended_selec_per = ['LoanID', 'MonthRep', 'CLDS','CAUPB']

df = pd.read_csv('Documents/3_ML4credit/troubleshooting/Data/Performance_HARP.txt', sep='|', names=col_per, dtype=perf_type_map, usecols=extended_selec_per, index_col=False,
                     nrows=None)  # set file path to the required text file


# In[5]:


# General information on the dataset

print("Dimensions of the dataset")
print(df.shape) # returns the dimensions of the dataset

print("Number of observations per month")
print(df.groupby('MonthRep').size())  # returns the number of observations per month

print("Number of observations per CLDS status")
print(df.groupby('CLDS').size())  # returns the number of observations per CLDS status

print("5 first and 5 last observations of the dataset")
print(df.head)


# In[6]:


df.dtypes # types of data in dataframe


# In[7]:


cat_columns = df.select_dtypes(['category']).columns
cat_columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
df
df.dtypes


# In[8]:


# Create list of sorted unique dates

uni_dates=list(set(df["MonthRep"]))

dates = [datetime.datetime.strptime(ts, "%m/%d/%Y") for ts in uni_dates]
dates.sort()
sorteddates = [datetime.datetime.strftime(ts, "%m/%d/%Y") for ts in dates]


# In[9]:


# This loop gives the amount of observations of each month
## Takes a while to run

month_obser = []
for i in range(0,len(sorteddates)):
    filter= df["MonthRep"]==sorteddates[i]
    temp=df[filter].shape[0]
    month_obser.append(temp)

# delete temporary varibels
del filter
del temp


# In[10]:


# Construct dataset that only displays the defaulted observations (goal is to observe all the ones in default)

default_filter= df["CLDS"]>2
default_df= df[default_filter]


print("Dimensions of the dataset containing only defaults")
print(default_df.shape)


# In[11]:


# this loop gives the amount of defaulted observations per month

month_default = []
for i in range(0,len(sorteddates)):
    filter= default_df["MonthRep"]==sorteddates[i]
    temp=default_df[filter].shape[0]
    month_default.append(temp)

# delete temporary varibels
del filter
del temp


# In[12]:


# this loop constructs the % monthly defaults on total monthly observations

default_rate =[]
for i in range(0,len(sorteddates)):
    temp= (month_default[i]/month_obser[i])*100
    default_rate.append(temp)

# delete temporary varibels
del temp


# In[14]:


# Plot of the default rate
register_matplotlib_converters()

sns.set()
sns.set_context("paper", font_scale=2.5)
sns.set_style("darkgrid", {'font.family': ['EYInterstate']})


# Plot of the default rate over time
fig, ax1 = plt.subplots()

ax1.plot(dates,default_rate, '-', color="darkblue")
ax1.set_xlabel("Year (HARP raw data)")
ax1.set_ylabel("Default rate (%)")

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(dates, month_obser, color="green")
ax2.set_ylabel("N° observed", color="green")
ax2.tick_params(axis='y', labelcolor="green")

plt.title("Default rate over time")
plt.rcParams["figure.figsize"]=30,15
plt.show()


# In[15]:


# This loop gives the total monthly CAUPB (current acutal unpaid balance) of all observed laons
## Takes a while to run

sum_month = []
for i in range(0,len(sorteddates)):
    filter= df[df["MonthRep"]==sorteddates[i]]
    temp=np.nansum(filter['CAUPB'])
    sum_month.append(temp)

# delete temporary variables
del filter
del temp


# In[16]:


# this loop gives the total montly CAUPB (current actual unpaid balance) of all defaulted loans

sum_default = []
for i in range(0,len(sorteddates)):
    filter= default_df[default_df["MonthRep"]==sorteddates[i]]
    temp= np.nansum(filter["CAUPB"])
    sum_default.append(temp)

# delete temporary variables
del filter
del temp


# In[17]:


# this loop constructs the % monthly defaults on total monthly observations

exposure_rate =[]
for i in range(0,len(sorteddates)):
    temp= (sum_default[i]/sum_month[i])*100
    exposure_rate.append(temp)

# delete temporary varibels
del temp

# warning stems from missing values first 2 months no CAUPB values
# the following 4 months there are only CAUPB values for defaulted loans, hence the observed 100%
# Missing data problem is resolved starting from nonth 7


# In[18]:


# Plot of the exposure rate

sns.set()
sns.set_context("paper", font_scale=2.5)
sns.set_style("darkgrid", {'font.family': ['EYInterstate']})


# Plot of the default rate over time
fig, ax1 = plt.subplots()

ax1.plot(dates,exposure_rate, '-', color="darkblue")
ax1.set_xlabel("Year (HARP raw data)")
ax1.set_ylabel("Exposure rate (%)")
ax1.set (ylim=(0,5))  # standard setting overruled

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.plot(dates, sum_month, color="green")
ax2.set_ylabel("Total exposure (USD)", color="green")
ax2.tick_params(axis='y', labelcolor="green")

plt.title("Exposure rate over time")
plt.rcParams["figure.figsize"]=30,15
plt.show()


# In[20]:


# Create stats on the performing loans
# Amount of the performing loans

month_performing =[]
for i in range(0,len(sorteddates)):
    temp= month_obser[i]-month_default[i]
    month_performing.append(temp)

# delete temporary varibels
del temp

# Exposure of the performing loans
sum_performing =[]
for i in range(0,len(sorteddates)):
    temp= sum_month[i]-sum_default[i]
    sum_performing.append(temp)

# delete temporary varibels
del temp



# In[28]:


df_results


# In[27]:


# writing the variables into one dataframe
df_results= pd.DataFrame()
df_results['Date'], df_results['# total observations'], df_results['# performing observations'] ,df_results['# defaulted observations'], df_results['Default rate %'], df_results['Exposure total observations USD'], df_results['Exposure performing observations USD'] ,df_results['Exposure defaulted observations USD'], df_results['Exposure rate'],  =[sorteddates, month_obser, month_performing ,month_default, default_rate, sum_month, sum_performing ,sum_default, exposure_rate]


# In[29]:


# Writing the dataframe to Excel
df_results.to_excel("Exploration HARP dataset output.xlsx")

