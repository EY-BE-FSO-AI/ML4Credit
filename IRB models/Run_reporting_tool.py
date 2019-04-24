#########################################################################################################################
###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibility of loading in py scripts without having to declare import statements twice
###          -Think about the treatment of 2.2-2.4
########################################################################################################################
###Import libraries###
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import sys
import os
import random

### Define local directory ###
local_dr  = os.path.normpath(os.path.expanduser("~/Documents/Python/GitHub/ML4Credit/IRB models"))
local_dr2 = os.path.normpath(os.path.expanduser("~/Documents/Python/"))
### Add local directory ###
sys.path.append(local_dr)

### Create development and monitoring data set ###
# Cross check with 2.5.1: Specific definitions.
from create_data_set import *

first_monitoring_year = datetime.date(2015, 1, 1)
df = pd.read_csv(local_dr2 + "/loan.csv", low_memory=False)
development_set, monitoring_set = data_model(data=df, ldate=first_monitoring_year).split_data()

### Probability of default (2.5)
### Add PD variable###
from model import *

FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'Default_Binary'
development_set, monitoring_set = model().PD_model(FEATURES, LABEL, development_set, monitoring_set, 'PD')

### #development_set.LGD_realised.hist()
development_set.LGD_realised.hist()
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'LGD_realised'
development_set, monitoring_set = model().LGD_model(FEATURES, LABEL, development_set, monitoring_set, 'LGD')

#CCF model
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'CCF_realised'
development_set, monitoring_set = model().CCF_model(FEATURES, LABEL, development_set, monitoring_set, 'CCF')
development_set.CCF.hist()

###Bin PDs
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'PD', 'Default_Binary', 75)
###Bin CCF
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'CCF', 'CCF_realised', 75)
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'CCF_realised', 'CCF_realised', 75)
### Bin LGD
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'LGD', 'LGD_realised', 75)

### LGD Test
### To be continued...

### Convert Rating grade into numberse###
monitoring_set.grade_num = monitoring_set.grade.apply(
    lambda x: {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}[x])
development_set.grade_num = development_set.grade.apply(
    lambda x: {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}[x])

### Test PD model ###

from Validation_tests import *

### Qualitative validation tools (2.5.2) (on hold, not a priority) ###
### Rating process statistics (2.5.2.1)
### Occurrence of overrides (2.5.2.2)
### Occurence of technical defaults (2.5.2.3)

# Define validation set

### Predictve ability (2.5.3)
### PD back-testing using a Jeffreys test (2.5.3.1)
# returns a dataframe with p-val column
# original exposure at the beginning of the period should still be added.

jeffrey_test = PD_tests().Jeffrey(development_set)

### Discriminatory power test - AUC (2.5.4)
### Current AUC vs AUC at initial validation/development (2.5.4.1)

validation_year = datetime.date(2016, 1, 1)
AUC_validation_year, s = PD_tests().AUC(monitoring_set.Default_Binary[(monitoring_set.issue_dt > validation_year) | (monitoring_set.Default_date > validation_year)],
                                        monitoring_set.grade_num[(monitoring_set.issue_dt > validation_year) | (monitoring_set.Default_date > validation_year)], 1)
AUC_development = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 0)[0]
AUC_S = (AUC_development - AUC_validation_year) / s
AUC_p = norm.pdf(AUC_S)
AUC_dev_years = []
for x in range(2007, 2014):
    AUC_dev_years.append(PD_tests().AUC(monitoring_set.Default_Binary[(monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (monitoring_set.Default_date.astype("datetime64[ns]").dt.year == x)],
                                        monitoring_set.grade_num[(monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (monitoring_set.Default_date.astype("datetime64[ns]").dt.year == x)], 0)[0])
AUC_bootstrap = []
random.seed = 1
for x in range(10000):
    sample = random.sample(range(len(development_set['Default_Binary'])), 10000)
    AUC_bootstrap.append(PD_tests().AUC(development_set.Default_Binary.iloc[sample], development_set.grade_num.iloc[sample], 0)[0])

plt.boxplot(AUC_bootstrap)

### Stability (2.5.5)

# Excluding defaulting customers
transition_matrix        = development_set[development_set.Default_Binary == 0].groupby("grade").Bin_PD.value_counts().unstack().fillna(0)
transition_matrix_freq   = transition_matrix / transition_matrix.sum(axis=0)
n_i                      = transition_matrix.sum(axis=1)

### Customer migrations (2.5.5.1)
# To be developped
# Create YYYY_rating column with a rating for each facility for each year

upper_MWB, lower_MWB = PD_tests().MWB(transition_matrix, transition_matrix_freq)

### Stability of migration matrix (2.5.5.2)
# To be developped

### Concentration in rating grades (2.5.5.3)
# calculate coefficient of variation and the herfindahl index
# p-val still needs to be calculated
CV, HI, CV_p_val = PD_tests().Herfindahl(development_set)

### Loss given default (2.6)
# To be developed
# Priority upon completion of PD

### Expected loss best estimate (2.7)
# To be developed

### LGD in-default (2.8)
# To be developped

### Credit conversion factor (2.9)
### Predictive ability (2.9.3)
### CCF back-testing using a t-test (2.9.3.1)

CCF_pval = CCF_tests().backtesting(development_set)


### Discriminatory power (2.9.4) 

gAUC_data_CCF = development_set[['Bin_CCF', 'Bin_CCF_realised']]

CCF_transition_matrix = gAUC_data_CCF.groupby('Bin_CCF_predicted').CCF_realised_grade.value_counts().unstack().fillna(0)
CCF_transition_matrix = CCF_transition_matrix.sort_index(axis=0)

transition_matrix_freq = transition_matrix / transition_matrix.sum(axis=0)
n_i = transition_matrix.sum(axis=1)



### gAUC (2.9.4.1)
CCF_gAUC = CCF_tests().gAUC(gAUC_data_CCF)



### Slotting approach for specialised lending exposures
# To be developped