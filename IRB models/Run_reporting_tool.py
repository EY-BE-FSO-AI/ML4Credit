<<<<<<< HEAD
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
import itertools
from sklearn import preprocessing
from scipy.stats import norm
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
import sys
import os
import random

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

### Define local directory ###
local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python"))
### Add local directory ###
sys.path.append(local_dr)

### Create development and monitoring data set ###
# Cross check with 2.5.1: Specific definitions.
from create_data_set import *

first_monitoring_year = datetime.date(2015, 1, 1)
df = pd.read_csv(local_dr + "/loan.csv", low_memory=False)
development_set, monitoring_set = data_model(data=df, ldate=first_monitoring_year).split_data()

### Probability of default (2.5)
### Add PD variable###
from model import *

FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'Default_Binary'
development_set, monitoring_set = model().PD_model(FEATURES, LABEL, development_set, monitoring_set, 'PD')

### Overwrite LGD until fix formula LGD
def LGD_star(df):
    star = 1 - ((1 + 0.05) - (1 + df.int_rate/100)* (1-df.PD)) / ((1 + df.int_rate/100)*df.PD)
    #star = np.minimum(1, np.maximum(0,development_set["LGD_realised"].values))
    star = np.minimum(1, np.maximum(0, star.values))
    return star

development_set["LGD_realised"] = LGD_star(development_set)
monitoring_set["LGD_realised"] = LGD_star(monitoring_set)
### #development_set.LGD_realised.hist()
development_set.LGD_realised.hist()
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'LGD_realised'
development_set, monitoring_set = model().LGD_model(FEATURES, LABEL, development_set, monitoring_set, 'LGD_predicted')

#CCF model
development_set["CCF_realised"] = development_set['revol_util'].fillna(100)
monitoring_set["CCF_realised"] = monitoring_set['revol_util'].fillna(100) 
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'CCF_realised'
development_set, monitoring_set = model().CCF_model(FEATURES, LABEL, development_set, monitoring_set, 'CCF_predicted')
#development_set.CCF_predicted.hist()
development_set.CCF_predicted.hist()

### Clusters of LGD
development_set["lgd_q"] = pd.qcut(development_set.LGD_predicted, q = 40, labels=np.arange(0,40))
development_set["Dflt_year"] = development_set.Default_date.astype("datetime64[ns]").dt.year
Data_q = development_set[development_set.Default_Binary == 1].groupby(["lgd_q", "Dflt_year"]).LGD_predicted.mean()
Data_q.unstack().plot(marker='o', linestyle='None')
Data_q = Data_q.unstack().fillna( 0 )
model = KMeans(n_clusters=7)
model.fit(Data_q)
Data_q["cluster_num"] = model.labels_
grade_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
cluster_dict = Data_q.to_dict()["cluster_num"]
grade_map = {k: grade_dict[v] for k, v in cluster_dict.items()}  # {Pd percentile : cluster grade}
development_set["lgd_cluster_label"] = development_set.lgd_q.apply(lambda g: grade_map[g])

### LGD Test
### To be continued...



### Convert PD into rating grade###
monitoring_set.grade_num = monitoring_set.grade.apply(
    lambda x: {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}[x])
development_set.grade_num = development_set.grade.apply(
    lambda x: {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}[x])

visual = development_set[development_set.Default_Binary == 1].head(100)

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
AUC_validation_year, s = PD_tests().AUC(monitoring_set.Default_Binary[(monitoring_set.issue_dt > validation_year) | (
            monitoring_set.Default_date > validation_year)],
                                        monitoring_set.grade_num[(monitoring_set.issue_dt > validation_year) | (
                                                    monitoring_set.Default_date > validation_year)], 1)
AUC_development = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 0)[0]
AUC_S = (AUC_development - AUC_validation_year) / s
AUC_p = norm.pdf(AUC_S)
AUC_dev_years = []
for x in range(2007, 2014):
    AUC_dev_years.append(PD_tests().AUC(monitoring_set.Default_Binary[
                                            (monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (
                                                        monitoring_set.Default_date.astype(
                                                            "datetime64[ns]").dt.year == x)],
                                        monitoring_set.grade_num[
                                            (monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (
                                                        monitoring_set.Default_date.astype(
                                                            "datetime64[ns]").dt.year == x)], 0)[0])
AUC_bootstrap = []
random.seed = 1
for x in range(10000):
    sample = random.sample(range(len(development_set['Default_Binary'])), 10000)
    AUC_bootstrap.append(
        PD_tests().AUC(development_set.Default_Binary.iloc[sample], development_set.grade_num.iloc[sample], 0)[0])

plt.boxplot(AUC_bootstrap)

### Stability (2.5.5)

development_set["Dflt_year"] = development_set.Default_date.astype("datetime64[ns]").dt.year
# Select relevant data
development_set["pd_q"] = pd.qcut(development_set.PD.dropna(), q=40, labels=np.arange(0, 40))
Data = development_set[["PD", "Default_Binary", "Dflt_year", "pd_q"]].dropna()
Data_q = Data.groupby(["pd_q", "Dflt_year"]).Default_Binary.mean()
Data_q.unstack().plot(marker='o',
                      linestyle='None')  # Or Data_q.unstack().plot(stacked=True, marker='o', linestyle='None')

# KMeans
# --> Attention nan values in 2007 for some deciles:
Data_q = Data_q.unstack().fillna(0)
Data_q.plot(marker='o', linestyle='None')
# Visually
model = KMeans(n_clusters=7)
model.fit(Data_q)
Data_q["cluster_num"] = model.labels_
grade_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
cluster_dict = Data_q.to_dict()["cluster_num"]
grade_map = {k: grade_dict[v] for k, v in cluster_dict.items()}  # {Pd percentile : cluster grade}
development_set["cluster_label"] = development_set.pd_q.apply(lambda g: grade_map[g])

development_set.groupby('grade').agg({'PD' : ['min','max']})

# Excluding defaulting customers
transition_matrix = development_set[development_set.Default_Binary == 0].groupby(
    "grade").cluster_label.value_counts().unstack().fillna(0)
transition_matrix_freq = transition_matrix / transition_matrix.sum(axis=0)
n_i = transition_matrix.sum(axis=1)

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
#replace linspace bin edges by KMEANS binning. This setup is illustrative.
development_set["CCF_predicted_grade"] = pd.cut(x = development_set['CCF_predicted'], bins = np.linspace(0,100,8), right = False, labels={'A', 'B', 'C', 'D', 'E', 'F', 'G'}) 
development_set["CCF_realised_grade"] = pd.cut(x = development_set['CCF_realised'], bins = np.linspace(0,100,8), right = False, labels={'A', 'B', 'C', 'D', 'E', 'F', 'G'})
#development_set["CCF_predicted_grade"] = pd.cut(x = development_set['CCF_predicted'], bins = np.linspace(0,100,8), right = False, labels={1, 2, 3, 4, 5, 6, 7}) 
#development_set["CCF_realised_grade"] = pd.cut(x = development_set['CCF_realised'], bins = np.linspace(0,100,8), right = False, labels={1, 2, 3, 4, 5, 6, 7})
gAUC_data_CCF = development_set[['CCF_predicted_grade', 'CCF_realised_grade']]

CCF_transition_matrix = gAUC_data_CCF.groupby('CCF_predicted_grade').CCF_realised_grade.value_counts().unstack().fillna(0)
CCF_transition_matrix = CCF_transition_matrix.sort_index(axis=0)

transition_matrix_freq = transition_matrix / transition_matrix.sum(axis=0)
n_i = transition_matrix.sum(axis=1)



### gAUC (2.9.4.1)
CCF_gAUC = CCF_tests().gAUC(gAUC_data_CCF)



### Slotting approach for specialised lending exposures
# To be developped



=======
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
import itertools
from sklearn import preprocessing
from scipy.stats import norm
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat
import sys
import os
import random

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

### Define local directory ###
local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python"))
### Add local directory ###
sys.path.append(local_dr)

### Create development and monitoring data set ###
# Cross check with 2.5.1: Specific definitions.
from create_data_set import *

first_monitoring_year = datetime.date(2015, 1, 1)
df = pd.read_csv(local_dr + "/loan.csv", low_memory=False)
development_set, monitoring_set = data_model(data=df, ldate=first_monitoring_year).split_data()

### Probability of default (2.5)
### Add PD variable###
from model import *

FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'Default_Binary'
development_set, monitoring_set = model().PD_model(FEATURES, LABEL, development_set, monitoring_set, 'PD')

### Overwrite LGD until fix formula LGD
def LGD_star(df):
    star = 1 - ((1 + 0.05) - (1 + df.int_rate/100)* (1-df.PD)) / ((1 + df.int_rate/100)*df.PD)
    #star = np.minimum(1, np.maximum(0,development_set["LGD_realised"].values))
    star = np.minimum(1, np.maximum(0, star.values))
    return star

development_set["LGD_realised"] = LGD_star(development_set)
monitoring_set["LGD_realised"] = LGD_star(monitoring_set)
### #development_set.LGD_realised.hist()
development_set.LGD_realised.hist()
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'LGD_realised'
development_set, monitoring_set = model().LGD_model(FEATURES, LABEL, development_set, monitoring_set, 'LGD_predicted')

#CCF model
development_set["CCF_realised"] = development_set['revol_util'].fillna(100)
monitoring_set["CCF_realised"] = monitoring_set['revol_util'].fillna(100) 
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
            'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
LABEL = 'CCF_realised'
development_set, monitoring_set = model().CCF_model(FEATURES, LABEL, development_set, monitoring_set, 'CCF_predicted')
#development_set.CCF_predicted.hist()
development_set.CCF_predicted.hist()

### Clusters of LGD
development_set["lgd_q"] = pd.qcut(development_set.LGD_predicted, q = 40, labels=np.arange(0,40))
development_set["Dflt_year"] = development_set.Default_date.astype("datetime64[ns]").dt.year
Data_q = development_set[development_set.Default_Binary == 1].groupby(["lgd_q", "Dflt_year"]).LGD_predicted.mean()
Data_q.unstack().plot(marker='o', linestyle='None')
Data_q = Data_q.unstack().fillna( 0 )
model = KMeans(n_clusters=7)
model.fit(Data_q)
Data_q["cluster_num"] = model.labels_
grade_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
cluster_dict = Data_q.to_dict()["cluster_num"]
grade_map = {k: grade_dict[v] for k, v in cluster_dict.items()}  # {Pd percentile : cluster grade}
development_set["lgd_cluster_label"] = development_set.lgd_q.apply(lambda g: grade_map[g])

### LGD Test
### To be continued...



### Convert PD into rating grade###
monitoring_set.grade_num = monitoring_set.grade.apply(
    lambda x: {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}[x])
development_set.grade_num = development_set.grade.apply(
    lambda x: {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}[x])

visual = development_set[development_set.Default_Binary == 1].head(100)

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
AUC_validation_year, s = PD_tests().AUC(monitoring_set.Default_Binary[(monitoring_set.issue_dt > validation_year) | (
            monitoring_set.Default_date > validation_year)],
                                        monitoring_set.grade_num[(monitoring_set.issue_dt > validation_year) | (
                                                    monitoring_set.Default_date > validation_year)], 1)
AUC_development = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 0)[0]
AUC_S = (AUC_development - AUC_validation_year) / s
AUC_p = norm.pdf(AUC_S)
AUC_dev_years = []
for x in range(2007, 2014):
    AUC_dev_years.append(PD_tests().AUC(monitoring_set.Default_Binary[
                                            (monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (
                                                        monitoring_set.Default_date.astype(
                                                            "datetime64[ns]").dt.year == x)],
                                        monitoring_set.grade_num[
                                            (monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (
                                                        monitoring_set.Default_date.astype(
                                                            "datetime64[ns]").dt.year == x)], 0)[0])
AUC_bootstrap = []
random.seed = 1
for x in range(10000):
    sample = random.sample(range(len(development_set['Default_Binary'])), 10000)
    AUC_bootstrap.append(
        PD_tests().AUC(development_set.Default_Binary.iloc[sample], development_set.grade_num.iloc[sample], 0)[0])

plt.boxplot(AUC_bootstrap)

### Stability (2.5.5)

development_set["Dflt_year"] = development_set.Default_date.astype("datetime64[ns]").dt.year
# Select relevant data
development_set["pd_q"] = pd.qcut(development_set.PD.dropna(), q=40, labels=np.arange(0, 40))
Data = development_set[["PD", "Default_Binary", "Dflt_year", "pd_q"]].dropna()
Data_q = Data.groupby(["pd_q", "Dflt_year"]).Default_Binary.mean()
Data_q.unstack().plot(marker='o',
                      linestyle='None')  # Or Data_q.unstack().plot(stacked=True, marker='o', linestyle='None')

# KMeans
# --> Attention nan values in 2007 for some deciles:
Data_q = Data_q.unstack().fillna(0)
Data_q.plot(marker='o', linestyle='None')
# Visually
model = KMeans(n_clusters=7)
model.fit(Data_q)
Data_q["cluster_num"] = model.labels_
grade_dict = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}
cluster_dict = Data_q.to_dict()["cluster_num"]
grade_map = {k: grade_dict[v] for k, v in cluster_dict.items()}  # {Pd percentile : cluster grade}
development_set["cluster_label"] = development_set.pd_q.apply(lambda g: grade_map[g])

development_set.groupby('grade').agg({'PD' : ['min','max']})

# Excluding defaulting customers
transition_matrix = development_set[development_set.Default_Binary == 0].groupby(
    "grade").cluster_label.value_counts().unstack().fillna(0)
transition_matrix_freq = transition_matrix / transition_matrix.sum(axis=0)
n_i = transition_matrix.sum(axis=1)

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
#replace linspace bin edges by KMEANS binning. This setup is illustrative.
development_set["CCF_predicted_grade"] = pd.cut(x = development_set['CCF_predicted'], bins = np.linspace(0,100,8), right = False, labels={'A', 'B', 'C', 'D', 'E', 'F', 'G'}) 
development_set["CCF_realised_grade"] = pd.cut(x = development_set['CCF_realised'], bins = np.linspace(0,100,8), right = False, labels={'A', 'B', 'C', 'D', 'E', 'F', 'G'})
#development_set["CCF_predicted_grade"] = pd.cut(x = development_set['CCF_predicted'], bins = np.linspace(0,100,8), right = False, labels={1, 2, 3, 4, 5, 6, 7}) 
#development_set["CCF_realised_grade"] = pd.cut(x = development_set['CCF_realised'], bins = np.linspace(0,100,8), right = False, labels={1, 2, 3, 4, 5, 6, 7})
gAUC_data_CCF = development_set[['CCF_predicted_grade', 'CCF_realised_grade']]

CCF_transition_matrix = gAUC_data_CCF.groupby('CCF_predicted_grade').CCF_realised_grade.value_counts().unstack().fillna(0)
CCF_transition_matrix = CCF_transition_matrix.sort_index(axis=0)

transition_matrix_freq = transition_matrix / transition_matrix.sum(axis=0)
n_i = transition_matrix.sum(axis=1)



### gAUC (2.9.4.1)
CCF_gAUC = CCF_tests().gAUC(gAUC_data_CCF)



### Slotting approach for specialised lending exposures
# To be developped



>>>>>>> 61e7df47bfe7c2bdcd9c7c9ddc0c419ea6501af5
