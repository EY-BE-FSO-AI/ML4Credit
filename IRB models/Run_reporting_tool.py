#########################################################################################################################
###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibility of loading in py scripts without having to declare import statements twice
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
###Define local directory###
local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python"))
###Add local directory###
sys.path.append(local_dr) 
###Create development and monitoring data set###
from create_data_set import *
first_monitoring_year = datetime.date(2015, 1, 1)
df = pd.read_csv(local_dr + "/loan.csv", low_memory=False)
development_set, monitoring_set =data_model(data = df, ldate= first_monitoring_year).split_data()
###Add PD variable###
from model import *
FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num',
           'funded_amnt_scaled','int_rate_scaled','inq_last_6mths_scaled','Income2TB_scaled'] 
LABEL = 'Default_Binary'
development_set, monitoring_set= model().PD_model(FEATURES, LABEL, development_set, monitoring_set, 'PD')
###Convert PD into rating grade###
monitoring_set.grade_num = monitoring_set.grade.apply(lambda x : {'A':7, 'B':6, 'C':5, 'D':4,'E':3,'F':2,'G':1}[x])
development_set.grade_num = development_set.grade.apply(lambda x : {'A':7, 'B':6, 'C':5, 'D':4,'E':3,'F':2,'G':1}[x])
###Test PD model###
###Discriminatory power test - AUC ###
from Validation_tests import *
validation_year = datetime.date(2016, 1, 1)
AUC_validation_year, s = PD_tests().AUC(   monitoring_set.Default_Binary[(monitoring_set.issue_dt > validation_year) | (monitoring_set.Default_date > validation_year)], 
                                        monitoring_set.grade_num     [(monitoring_set.issue_dt > validation_year) | (monitoring_set.Default_date > validation_year)], 1)
AUC_development = PD_tests().AUC(development_set.Default_Binary, development_set.grade_num, 0)[0]
AUC_S = (AUC_development - AUC_validation_year) / s
AUC_p = norm.pdf(AUC_S)
AUC_dev_years = []
for x in range(2007,2014) : 
 AUC_dev_years.append(PD_tests().AUC(   monitoring_set.Default_Binary[(monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (monitoring_set.Default_date.astype("datetime64[ns]").dt.year == x)], 
                                        monitoring_set.grade_num     [(monitoring_set.issue_dt.astype("datetime64[ns]").dt.year == x) | (monitoring_set.Default_date.astype("datetime64[ns]").dt.year == x)], 0)[0])
AUC_bootstrap = []
random.seed = 1
for x in range(10000) :
     sample = random.sample(range(len(development_set['Default_Binary'])), 10000)
     AUC_bootstrap.append(PD_tests().AUC(development_set.Default_Binary.iloc[sample], development_set.grade_num.iloc[sample], 0)[0])

plt.boxplot(AUC_bootstrap)

###Jeffrey's test for PD backtesting ### - Please check implementation
#returns a dataframe with p-val column
jeffrey_test = PD_tests().Jeffrey(development_set)

### Concentration in rating grades (2.5.5.3)
#calculate coefficient of variation and the herfindahl index
CV, HI, CV_p_val = PD_tests().Herfindahl(development_set)