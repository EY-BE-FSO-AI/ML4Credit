# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 11:59:24 2019

@author: bebxadvsven
"""

""" README:
There are 25 variables:
ID: ID of each client
LIMIT_BAL: Amount of given credit in NT dollars (includes individual and family/supplementary credit
SEX: Gender (1=male, 2=female)
EDUCATION: (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
MARRIAGE: Marital status (1=married, 2=single, 3=others)
AGE: Age in years
PAY_0: Repayment status in September, 2005 (-1=pay duly, 1=payment delay for one month, 2=payment delay for two months, ... 8=payment delay for eight months, 9=payment delay for nine months and above)
PAY_2: Repayment status in August, 2005 (scale same as above)
PAY_3: Repayment status in July, 2005 (scale same as above)
PAY_4: Repayment status in June, 2005 (scale same as above)
PAY_5: Repayment status in May, 2005 (scale same as above)
PAY_6: Repayment status in April, 2005 (scale same as above)
BILL_AMT1: Amount of bill statement in September, 2005 (NT dollar)
BILL_AMT2: Amount of bill statement in August, 2005 (NT dollar)
BILL_AMT3: Amount of bill statement in July, 2005 (NT dollar)
BILL_AMT4: Amount of bill statement in June, 2005 (NT dollar)
BILL_AMT5: Amount of bill statement in May, 2005 (NT dollar)
BILL_AMT6: Amount of bill statement in April, 2005 (NT dollar)
PAY_AMT1: Amount of previous payment in September, 2005 (NT dollar)
PAY_AMT2: Amount of previous payment in August, 2005 (NT dollar)
PAY_AMT3: Amount of previous payment in July, 2005 (NT dollar)
PAY_AMT4: Amount of previous payment in June, 2005 (NT dollar)
PAY_AMT5: Amount of previous payment in May, 2005 (NT dollar)
PAY_AMT6: Amount of previous payment in April, 2005 (NT dollar)
default.payment.next.month: Default payment (1=yes, 0=no)
"""


### QUESTIONS: ###
#1: What is the meaning of PAY = '-2' anf of PAY = '0'?
# Victor: Explanations about the variables PAY?
#2: How do we know what was the outstanding amount before the observations in the dataset?
#3: What is the DoD -> Get default time -> construct drawn, undrawn and CCF at that time
# Victor: Defaulting only means not paying for the current month as there seems to be no correlation between the not paying for X months and defaulting?
# Victor: Is it normal that the amount paid is so low compared to the bills? Is it because the interests rates on credit cards are lower in Taiwan

### OBSERVATIONS: ###
#1: 't' for PAYAMT corresponds to 't-1' for BILLAMT
#2: PAY1 is not observed.

import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
local_dr = os.path.normpath(os.path.expanduser("~/Documents/Python/"))
data = pd.read_csv(local_dr + "/UCI_Credit_Card.csv", low_memory=False)
#data = data[data["PAY_0"]==3] #subsample of defaulted observations, 1.07% of original sample
data.default_flag = data["PAY_0"]==3  #Definition of default

#creation of due_amt, paid_amt, outstanding and utilization
data.ID.count() #30K unique customers

#exposure
data["exposure"] = data["BILL_AMT4"] + data["BILL_AMT5"] + data["BILL_AMT6"]

#exposure at default
data["exposure_def"] = data["BILL_AMT1"] + data["BILL_AMT2"] + data["BILL_AMT3"] 


#credit conversion factor
#consider looking at CCF only for defaulted exposures data.default_flag = True

data["CCF"] = (data.exposure_def  - data.exposure)/(data.LIMIT_BAL*3 - data.exposure)  

def var_bind(df, target_var, L_bound, U_bound):
    df[target_var] = np.minimum(U_bound, np.maximum(L_bound, df[target_var]))
    return df

var_bind(df = data, target_var = 'CCF', L_bound = 0, U_bound = math.inf)


#visualisation
#plt.scatter(data.index, data.CCF.apply(lambda x: np.NaN if x == 0 else math.log(x)))
plt.hist(data.CCF, bins='auto')

