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
#do we need to import pandas? import pandas as pd

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
#########################################################################################################################

#LGD model
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'LGD_realised'
development_set, monitoring_set = model().LGD_model(FEATURES, LABEL, development_set, monitoring_set, 'LGD')

#CCF model
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'CCF_realised'
development_set, monitoring_set = model().CCF_model(FEATURES, LABEL, development_set, monitoring_set, 'CCF')

FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num']
LABEL = 'CCF_realised'
development_set, monitoring_set = model().CCF_model(FEATURES, LABEL, development_set, monitoring_set, 'CCF_')

###Bin PDs
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'PD', 'Default_Binary', 75)
###Bin CCF
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'CCF',  'CCF_realised', 75)
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'CCF_', 'CCF_realised', 75)
### Bin LGD
development_set, monitoring_set = model().binning_monotonic(development_set, monitoring_set, 'LGD', 'LGD_realised', 75)

### LGD Test
### To be continued...

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
AUC_validation_year, s_curr = PD_tests().AUC(monitoring_set.Default_Binary[(monitoring_set.obs_dt > validation_year) | (monitoring_set.Default_date > validation_year)],
                                        monitoring_set.grade_num[(monitoring_set.obs_dt > validation_year) | (monitoring_set.Default_date > validation_year)], 1)
AUC_development, s_init = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 0)
AUC_S = (AUC_development - AUC_validation_year) / s_curr
AUC_p = norm.pdf(AUC_S)
AUC_dev_years = []
for x in range(2007, 2014):
    AUC_dev_years.append(PD_tests().AUC(monitoring_set.Default_Binary[(monitoring_set.obs_dt.astype("datetime64[ns]").dt.year == x) | (monitoring_set.Default_date.astype("datetime64[ns]").dt.year == x)],
                                        monitoring_set.grade_num[(monitoring_set.obs_dt.astype("datetime64[ns]").dt.year == x) | (monitoring_set.Default_date.astype("datetime64[ns]").dt.year == x)], 0)[0])
AUC_bootstrap = []
random.seed = 1
for x in range(10000):
    sample = random.sample(range(len(development_set['Default_Binary'])), 10000)
    AUC_bootstrap.append(PD_tests().AUC(development_set.Default_Binary.iloc[sample], development_set.grade_num.iloc[sample], 0)[0])

plt.boxplot(AUC_bootstrap)

### Stability (2.5.5)

# Excluding defaulting customers
transition_matrix       = development_set[development_set.Default_Binary == 0].groupby(['grade_num', 'Bin_PD']).size().unstack(fill_value=0)
transition_matrix_freq  = transition_matrix / transition_matrix.sum(axis=0)

### Customer migrations (2.5.5.1)
# To be developped
# Create YYYY_rating column with a rating for each facility for each year

upper_MWB, lower_MWB = PD_tests().MWB(transition_matrix, transition_matrix_freq)

### Stability of migration matrix (2.5.5.2)
z_up, z_low, zUP_pval, zDOWN_pval = PD_tests().stability_migration_matrix(transition_matrix, transition_matrix_freq)

### Concentration in rating grades (2.5.5.3)
# calculate coefficient of variation and the herfindahl index
K = len(development_set[development_set.Default_Binary == 0].grade.unique()) #number of rating grades for non-defaulted exposures
CV_init, HI_init, _ = PD_tests().Herfindahl(development_set)
CV_curr, HI_curr, _ = PD_tests().Herfindahl(monitoring_set)
cr_pval = 1 - norm.cdf(np.sqrt(K - 1) * (CV_curr - CV_init) / np.sqrt(CV_curr**2 * (0.5 + CV_curr**2)))


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""" "Loss Given Default" """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### Predictive Ability (2.6.2)
### LGD back-testing using a t-test (2.6.2.1)
LGD_backtesting_pval = LGD_tests().backtesting(development_set)

### Discriminatory Power (2.6.3)
### gAUC for LGD (2.6.3.1)
dev_LGD_transition_matrix        = development_set[development_set.Default_Binary == 0].groupby(['grade_num', 'Bin_LGD']).size().unstack(fill_value=0)
dev_LGD_transition_matrix_freq = dev_LGD_transition_matrix / dev_LGD_transition_matrix.sum(axis=0)
mon_LGD_transition_matrix        = monitoring_set[monitoring_set.Default_Binary == 0].groupby(['grade_num', 'Bin_LGD']).size().unstack(fill_value=0)
mon_LGD_transition_matrix_freq = mon_LGD_transition_matrix / mon_LGD_transition_matrix.sum(axis=0)

LGD_gAUC_init, LGD_gAUC_curr, LGD_S, LGD_p_val = LGD_tests().gAUC_LGD(mon_LGD_transition_matrix, dev_LGD_transition_matrix)

### LGD: Qualitative validation tools (2.6.4)
### Population Stability Index(2.6.4.2)
development_set.LGD_realised = development_set.LGD_realised.astype(float) #LGD_realised was stored as object
development_set.LGD = development_set.LGD.astype(float) #LGD_realised was stored as object
LGD_psi = LGD_tests().psi_lgd(data_set=development_set)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""" "Credit Conversion Factor" """"""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### Credit conversion factor (2.9)

### Predictive ability (2.9.3)
### CCF back-testing using a t-test (2.9.3.1)
CCF_backtesting_pval = CCF_tests().backtesting(development_set, 'CCF', 'CCF_')

### Discriminatory power (2.9.4)
### gAUC (2.9.4.1)
dev_CCF_transition_matrix        = development_set[development_set.Default_Binary == 0].groupby(['grade_num', 'Bin_CCF']).size().unstack(fill_value=0)
dev_CCF_transition_matrix_freq = dev_CCF_transition_matrix / dev_CCF_transition_matrix.sum(axis=0)
mon_CCF_transition_matrix        = monitoring_set[monitoring_set.Default_Binary == 0].groupby(['grade_num', 'Bin_CCF']).size().unstack(fill_value=0)
mon_CCF_transition_matrix_freq = mon_CCF_transition_matrix / mon_CCF_transition_matrix.sum(axis=0)

CCF_gAUC_init, CCF_gAUC_curr, CCF_S, CCF_p_val = CCF_tests().gAUC_CCF(mon_CCF_transition_matrix, dev_CCF_transition_matrix)
#gAUC_data_CCF = development_set[['Bin_CCF', 'Bin_CCF_']]
### Qualitative validation tools (2.9.5)
### Population Stability Index (2.9.5.1)
development_set.CCF = development_set.CCF.astype(float) #LGD_realised was stored as object
development_set.CCF_ = development_set.CCF_.astype(float) #LGD_realised was stored as object
CCF_psi = CCF_tests().psi_ccf(data_set=development_set)

### Slotting approach for specialised lending exposures (2.10)
# To be developed

### Export to Excel
from export import *

# Define some missing values (Ideally should be given in the testing results)
start_date = datetime.date(2007,1,1)
end_date = datetime.date(2015, 1, 1)
nb_customer = len(development_set.id.unique())
nb_rating_grades = len(development_set.grade.unique())
name_rating_grades = jeffrey_test.index.tolist()[:-1]
development_set.PD = development_set.PD.astype(float) #changing dtype of PD column to use groupby
averagePD_pergrade = development_set.groupby("grade").PD.mean().values
nb_customer_pergrade = development_set.grade.value_counts().sort_index().values
nb_default_pergrade = jeffrey_test[('Default_Binary', 'sum')].values[:-1]
development_set["original_exposure"] = (development_set.installment * development_set.term).astype(float)
original_exposure_pergrade = development_set.groupby("grade").original_exposure.sum().values
jeffrey_test_pval_pergrade = jeffrey_test.p_val.values[:-1]
s_init = 0

# Store eveything in dictionary
PD_excel_input = {
    "predictive_ability" : [name_rating_grades, averagePD_pergrade, nb_customer_pergrade, nb_default_pergrade,
                            jeffrey_test_pval_pergrade, original_exposure_pergrade],
    "AUC" : [AUC_development, AUC_validation_year, s_curr, AUC_S, AUC_p, "yes", start_date, end_date, nb_customer, s_init],
    "customer_migrations" : [upper_MWB, lower_MWB],
    "concentration_rating_grades" : [HI_init, HI_curr, cr_pval, 0, start_date, end_date, nb_customer,
                                     nb_rating_grades],
    "stability_migration_matrix" : [z_up, z_low, zUP_pval, zDOWN_pval],
}


export().PD_toExcel( PD_excel_input )

