###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibility of loading in py scripts without having to declare import statements twice
###          -Think about the treatment of 2.2-2.4
########################################################################################################################

### Define local directory ###
import os
import time
import numpy as np
local_dr  = os.path.normpath(os.path.expanduser("~/Documents/Python/GitHub/ML4Credit/IRB models"))
local_dr2 = os.path.normpath(os.path.expanduser("~/Documents/Python/"))
### Add local directory ###
import sys
sys.path.append(local_dr)
###Import data set###
from create_data_set import *
start_time = time.time()
development_set, validation_set = import_data().EY(local_dr2)
elapsed_time = time.time() - start_time
print('Data creation execution time: %.3fs' % (elapsed_time))
###Validation tests###
from Validation_tests import *
#####PD

### Portfolio information ###

[RWEA_dev, RWEA_val] = [0, 0]               #placeholder
[EAD_dev, EAD_val] = [0, 0]                 #placeholder
[PD_M_dev, PD_M_val] = [0, 0]               #link with 2.5.2 qualitative statistics
[PD_K_dev, PD_K_val] = [len(development_set.grade_num.unique()), len(validation_set.grade_num.unique())]               #link with number of rating grades calculation further down the code
[EV_default_dev, EV_default_val] = [0, 0]   #placeholder
[default_dev, default_val] = [0, 0]         #link with default calculations further down the code

####### Qualitative validation tools (2.5.2) (on hold, not a priority) ###

PD_M_ex_ORFS_FLAG = "no"            # "yes" if taken into account in data/model "no" if not
PD_M_ex_TR_FLAG = "no"              # "yes" if taken into account in data/model "no" if not
PD_M_def_overrides_FLAG = "no"      # "yes" if taken into account in data/model "no" if not
PD_M_def_technical_FLAG = "no"      # "yes" if taken into account in data/model "no" if not

####### Rating process statistics (2.5.2.1)
#push to create dataset script
PD_M = np.count_nonzero(development_set.id)

PD_M_ex_ORFS = 0        #=development_set.ORFS[development_set.ORFS==1].sum()
averagePD_M_ex_ORFS = 0 #=development_set.PD[development_set.ORFS==1].mean()
adfPD_M_ex_ORFS = 0     #=development_set.ORFS[development_set.Default_Binary==1].sum()

PD_M_ex_TR = 0          #=development_set.TR[development_set.TR==1].sum()
averagePD_M_ex_TR = 0   #=development_set.PD[development_set.TR==1].mean()
adfPD_M_ex_TR = 0       #=development_set.TR[development_set.Default_Binary==1].sum()

PD_M_EPD = 0            #count exclusion due to rating process deficieny

PD_N = PD_M - PD_M_ex_ORFS - PD_M_ex_TR - PD_M_EPD
####### Occurrence of overrides (2.5.2.2)

PD_M_def_overrides = 0  #=development_set.override[development_set.override==1].sum()

####### Occurence of technical defaults (2.5.2.3)

PD_M_def_technical = 0  #count exclusion due to technical defaults

####### Predictve ability (2.5.3)
jeffrey_test = PD_tests().Jeffrey(development_set[['grade', 'PD', 'Default_Binary']], 'grade', 'PD', 'Default_Binary')
####### Discriminatory power test - AUC (2.5.4)
start_time = time.time()
PD_AUC_val, PD_s_val = PD_tests().AUC(validation_set.Default_Binary, validation_set.grade_num, 1)
PD_AUC_dev, PD_s_dev = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 1)
PD_AUC_S             = general().test_stat(PD_AUC_dev, PD_AUC_val, PD_s_val)
PD_AUC_p             = general().p_value(PD_AUC_S)
elapsed_time = time.time() - start_time
print('PD AUC calculation execution time: %.3fs' % (elapsed_time))
#########Extra tests
start_time = time.time()
AUC_dev_years = extra_tests().range(development_set, 'out.append(PD_tests().AUC(df.Default_Binary,     df.grade_num,     0)[0])')
AUC_bootstrap = extra_tests().boot( development_set, 'out.append(PD_tests().AUC(sample.Default_Binary, sample.grade_num, 0)[0])', 10000, 20000)
elapsed_time = time.time() - start_time
print('PD extra tests calculation execution time: %.3fs' % (elapsed_time))
### Stability (2.5.5)
##### Excluding defaulting customers
start_time = time.time()
transition_matrix        = matrix().matrix_obs(development_set, 'grade_num', 'Bin_PD', 'Default_Binary')
transition_matrix_freq   = matrix().matrix_prob(transition_matrix)
elapsed_time = time.time() - start_time
### Customer migrations (2.5.5.1)
# Create YYYY_rating column with a rating for each facility for each year
upper_MWB, lower_MWB = PD_tests().mwb_(transition_matrix, transition_matrix_freq)
### Stability of migration matrix (2.5.5.2)
z, z_pval = PD_tests().stability_migration_matrix(transition_matrix, transition_matrix_freq)
### Concentration in rating grades (2.5.5.3)
# calculate coefficient of variation and the herfindahl index
CV_init, HI_init, CV_curr, HI_curr, cr_pval = PD_tests().Herfindahl(development_set[['grade', 'Default_Binary']], validation_set[['grade', 'Default_Binary']], 'grade', 'Default_Binary', 'grade', 'count')
CV_init_exp, HI_init_exp, CV_curr_exp, HI_curr_exp, cr_pval_exp = PD_tests().Herfindahl(development_set[['grade', 'Default_Binary', 'original_exposure']], validation_set[['grade', 'Default_Binary', 'original_exposure']], 'grade', 'Default_Binary', 'original_exposure', 'sum')
print('PD transition matrix stability test and rating grade concentration execution time: %.3fs' % (elapsed_time))

#####LGD
####### Predictive Ability (2.6.2)
####### LGD back-testing using a t-test (2.6.2.1)
LGD_backtesting_ptf = LGD_tests().backtesting(development_set, "LGD", "LGD_realised")
LGD_backtesting_perGrade = LGD_tests().backtesting_facilityGrade(development_set, "LGD", "LGD_realised")
####### Discriminatory Power (2.6.3)
####### gAUC for LGD (2.6.3.1)
dev_LGD_transition_matrix        = matrix().matrix_obs(development_set, 'Bin_LGD', 'Bin_LGD_realised', 'Default_Binary')
dev_LGD_transition_matrix_freq   = matrix().matrix_prob(dev_LGD_transition_matrix)
val_LGD_transition_matrix        = matrix().matrix_obs(validation_set, 'Bin_LGD', 'Bin_LGD_realised', 'Default_Binary')
val_LGD_transition_matrix_freq   = matrix().matrix_prob(val_LGD_transition_matrix)

LGD_gAUC_init, LGD_gAUC_curr, LGD_S, LGD_curr_var, LGD_init_var, LGD_p_val = LGD_tests().gAUC_LGD(val_LGD_transition_matrix_freq, dev_LGD_transition_matrix_freq)

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
CCF_backtesting_ptf = CCF_tests().backtesting(development_set, 'CCF', 'CCF_')
CCF_backtesting_perGrade = CCF_tests().backtesting_facilityGrade(development_set, 'CCF', 'CCF_')

### Discriminatory power (2.9.4)
### gAUC (2.9.4.1)
dev_CCF_transition_matrix        = matrix().matrix_obs(development_set, 'Bin_CCF', 'Bin_CCF_realised', 'Default_Binary')
dev_CCF_transition_matrix_freq   = matrix().matrix_prob(dev_CCF_transition_matrix)
val_CCF_transition_matrix        = matrix().matrix_obs(validation_set, 'Bin_CCF', 'Bin_CCF_realised', 'Default_Binary')
val_CCF_transition_matrix_freq   = matrix().matrix_prob(val_CCF_transition_matrix)

CCF_gAUC_init, CCF_gAUC_curr, CCF_S, CCF_curr_var, CCF_init_var, CCF_p_val = CCF_tests().gAUC_CCF(val_CCF_transition_matrix, dev_CCF_transition_matrix)
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

# Store eveything in dictionary
PD_excel_input = {
     "name"                         : "Demo PD.xlsx",
     "start"                        : datetime.date(2007, 1, 1),
     "end"                          : datetime.date(2015, 1, 1),
     "portfolio_information"        : {"RWEA_dev" : RWEA_dev, "RWEA_val":RWEA_val,"EAD_dev": EAD_dev,"EAD_val": EAD_val,
                                       "PD_M_dev": PD_M_dev,"PD_M_val": PD_M_val,"PD_K_dev": PD_K_dev, "PD_K_val": PD_K_val,
                                       "EV_default_dev":EV_default_dev, "EV_default_val":EV_default_val,
                                       "default_dev":default_dev, "default_val":default_val},
     "qualitative"                  : {"PD_M_ex_ORFS_FLAG": PD_M_ex_ORFS_FLAG, "PD_M_ex_TR_FLAG": PD_M_ex_TR_FLAG,
                                       "PD_M_def_overrides_FLAG":PD_M_def_overrides_FLAG, "PD_M_def_technical_FLAG":PD_M_def_technical_FLAG,
                                       "PD_M":PD_M, "PD_M_ex_ORFS":PD_M_ex_ORFS, "averagePD_M_ex_ORFS":averagePD_M_ex_ORFS,
                                       "adfPD_M_ex_ORFS":adfPD_M_ex_ORFS, "PD_M_ex_TR":PD_M_ex_TR,
                                       "averagePD_M_ex_TR":averagePD_M_ex_TR, "adfPD_M_ex_TR":adfPD_M_ex_TR, "PD_M_EPD":PD_M_EPD,
                                       "PD_N":PD_N, "PD_M_def_overrides":PD_M_def_overrides, "PD_M_def_technical":PD_M_def_technical},
     "jeffrey"                      : jeffrey_test,
     "AUC_init"                     : PD_s_dev,
     "AUC"                          : [PD_AUC_dev, PD_AUC_val, PD_s_val, PD_AUC_S, PD_AUC_p, "yes", 0, 0, 0, PD_s_dev],
     "customer_migrations"          : [upper_MWB, lower_MWB],
     "concentration_rating_grades"  : [HI_init, HI_curr, cr_pval, HI_init_exp],
     "stability_migration_matrix"   : [transition_matrix_freq, z, z_pval],
     "avg_PD"                       : development_set.groupby("grade").PD.mean().values,
     "nb_cust"                      : development_set.grade.value_counts().sort_index().values,
     "orgExp_Grade"                 : development_set.groupby("grade").original_exposure.sum().values,
}
export().PD_toExcel(PD_excel_input)

LGD_excel_inputs = {
    "predictive_ability": [LGD_backtesting_ptf, LGD_backtesting_perGrade],
    "AUC": [LGD_gAUC_init, LGD_gAUC_curr, LGD_S, LGD_curr_var, LGD_init_var, LGD_p_val],
    "stability_migration_matrix": [z_up, z_low, zUP_pval, zDOWN_pval],
}
export().LGD_toExcel(development_set, LGD_excel_inputs)

CCF_excel_inputs = {
    "predictive_ability": [CCF_backtesting_ptf, CCF_backtesting_perGrade],
    "AUC": [CCF_gAUC_init, CCF_gAUC_curr, CCF_S, CCF_curr_var, CCF_p_val, CCF_init_var],
    "stability_migration_matrix": [z_up, z_low, zUP_pval, zDOWN_pval],
}
export().CCF_toExcel(development_set, CCF_excel_inputs)




