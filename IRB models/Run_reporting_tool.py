###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibility of loading in py scripts without having to declare import statements twice
###          -Think about the treatment of 2.2-2.4
########################################################################################################################

### Define local directory ###
import os
local_dr  = os.path.normpath(os.path.expanduser("~/Documents/Python/GitHub/ML4Credit/IRB models"))
local_dr2 = os.path.normpath(os.path.expanduser("~/Documents/Python/"))
### Add local directory ###
import sys
sys.path.append(local_dr)
###Import data set###
from create_data_set import *
development_set, validation_set = import_data().EY(local_dr2)
###Validation tests###
from Validation_tests import *
#####PD
####### Qualitative validation tools (2.5.2) (on hold, not a priority) ###
####### Rating process statistics (2.5.2.1)
####### Occurrence of overrides (2.5.2.2)
####### Occurence of technical defaults (2.5.2.3)
####### Predictve ability (2.5.3)
jeffrey_test = PD_tests().Jeffrey(development_set)
####### Discriminatory power test - AUC (2.5.4)
PD_AUC_val, PD_s_val = PD_tests().AUC(validation_set.Default_Binary, validation_set.grade_num, 1)
PD_AUC_dev           = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 0)
PD_AUC_S             = general().test_stat(PD_AUC_dev, PD_AUC_val, PD_s_val)
PD_AUC_p             = general().p_value(PD_AUC_S)
#########Extra tests
AUC_dev_years = extra_tests().range(development_set, 'out.append(PD_tests().AUC(df.Default_Binary,     df.grade_num,     0)[0])')
AUC_bootstrap = extra_tests().boot( development_set, 'out.append(PD_tests().AUC(sample.Default_Binary, sample.grade_num, 0)[0])', 10000, 20000)
### Stability (2.5.5)
##### Excluding defaulting customers
transition_matrix        = matrix().matrix_obs(development_set, 'grade_num', 'Bin_PD', 'Default_Binary')
transition_matrix_freq   = matrix().matrix_prob(transition_matrix)
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
CV_curr, HI_curr, _ = PD_tests().Herfindahl(validation)
cr_pval = 1 - norm.cdf(np.sqrt(K - 1) * (CV_curr - CV_init) / np.sqrt(CV_curr**2 * (0.5 + CV_curr**2)))


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""" "Loss Given Default" """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

### Predictive Ability (2.6.2)
### LGD back-testing using a t-test (2.6.2.1)
LGD_backtesting_pval = LGD_tests().backtesting(development_set)

### Discriminatory Power (2.6.3)
### gAUC for LGD (2.6.3.1)
dev_LGD_transition_matrix        = matrix().matrix_obs(development_set, 'Bin_LGD', 'Bin_LGD_realised', 'Default_Binary')
dev_LGD_transition_matrix_freq   = matrix().matrix_prob(dev_LGD_transition_matrix)
val_LGD_transition_matrix        = matrix().matrix_obs(validation_set, 'Bin_LGD', 'Bin_LGD_realised', 'Default_Binary')
val_LGD_transition_matrix_freq   = matrix().matrix_prob(val_LGD_transition_matrix)

LGD_gAUC_init, LGD_gAUC_curr, LGD_S, LGD_p_val = LGD_tests().gAUC_LGD(val_LGD_transition_matrix_freq, dev_LGD_transition_matrix_freq)

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
dev_CCF_transition_matrix        = matrix().matrix_obs(development_set, 'Bin_CCF', 'Bin_CCF_realised', 'Default_Binary')
dev_CCF_transition_matrix_freq   = matrix().matrix_prob(dev_CCF_transition_matrix)
val_CCF_transition_matrix        = matrix().matrix_obs(validation, 'Bin_CCF', 'Bin_CCF_realised', 'Default_Binary')
val_CCF_transition_matrix_freq   = matrix().matrix_prob(val_CCF_transition_matrix)

CCF_gAUC_init, CCF_gAUC_curr, CCF_S, CCF_p_val = CCF_tests().gAUC_CCF(val_CCF_transition_matrix, dev_CCF_transition_matrix)
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