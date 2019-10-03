import os
import pandas                      as pd
import numpy                       as np
import seaborn                     as sns
import matplotlib.pyplot           as plt
from sklearn.metrics      import roc_curve, auc
from scipy.stats          import beta
from scipy.stats          import norm
from scipy.stats          import binom
from scipy.stats          import t

# Import ML output
df_ML                    = pd.read_csv(os.getcwd()+'\predictions_train.csv').iloc[:, 1:]
df_ML['training']        = 1
df_dummy_ML              = pd.read_csv(os.getcwd()+'\predictions_validation.csv').iloc[:, 1:]
df_dummy_ML['training']  = 0
df_ML                    = df_ML.append(df_dummy_ML).sort_values(by='LoanID').reset_index().iloc[:, 1:]

# Import Classical output
df_CL                    = pd.read_csv(os.getcwd()+'\predictions_train_cl.csv')[['LoanID','Default','Prediction']]
df_CL['training']        = 1
df_dummy_CL              = pd.read_csv(os.getcwd()+'\predictions_validation_cl.csv')[['LoanID','Default','Prediction']]
df_dummy_CL['training']  = 0
df_CL                    = df_CL.append(df_dummy_CL).sort_values(by='LoanID').reset_index().iloc[:, 1:]
df_CL                    = df_CL.rename(columns={"Default": "default", "Prediction": "pb_default"})
df_CL                    = df_CL[['LoanID','pb_default','training','default']] #Re-arrange to match ML df.

# Add default flag to ML df
df_ML                    = pd.merge(df_ML,df_CL[['LoanID','default']], on='LoanID', how='inner')

# Master Scale
ratings            = 22                                                                           #Number of rating grades
PD_min             = 0.0003                                                                       #Minimum PD value (regulatory threshold)
slope              = (np.log((2-PD_min)/PD_min)-0)/(ratings-1)                                    #Slope between min PD value and default in logit space, depending on NB of rating grades
MS                 = 2/(1+np.exp(slope*pd.Series(list(range(ratings)))))
idx                = pd.IntervalIndex.from_arrays(MS[1:].append(pd.Series(0)), MS, closed='left')
df_ML['rating_PD'] = MS[idx.get_indexer(df_ML.pb_default)].values
df_CL['rating_PD'] = MS[idx.get_indexer(df_CL.pb_default)].values

# Validation test - Accuracy
class CL_PD_tests(object):      
     ###Jeffrey's Test
     def Jeffrey(self, df, x, y , z):
          #df is a dataframe object that contains columns x, y and z
          #x aggregation variable (rating grade for PD test)
          #y modelled variable
          #z observed variable (expected to be binary)
          #alpha = D + 1/2
          #beta = Nc- D + 1/2
          aggregation                                 = df.groupby(x).agg({x:'count', y: ['sum', 'count', 'mean'], z: ['sum', 'count', 'mean']})
          aggregation['Observed']                     = aggregation[(z, 'mean')]
          aggregation['alpha']                        = aggregation[(z, 'sum')] + 1/2
          aggregation['beta']                         = aggregation[(z, 'count')] - aggregation['alpha'] + 1
          aggregation['H0']                           = aggregation[(y, 'mean')]
          aggregation['p_val']                        = beta.cdf(aggregation['H0'], aggregation['alpha'], aggregation['beta'])
          aggregation.loc['Portfolio']                = aggregation.sum()
          aggregation[y, 'mean'].loc['Portfolio']     = df.agg({y: 'mean'}).values
          aggregation[z, 'mean'].loc['Portfolio']     = df.agg({z: 'mean'}).values
          aggregation['Observed'].loc['Portfolio']    = df.agg({z: 'mean'}).values
          aggregation['alpha'].loc['Portfolio']       = df.agg({z: 'sum' }).values + 1/2
          aggregation['beta'].loc['Portfolio']        = df.agg({z: 'count'}).values - aggregation['alpha'].loc['Portfolio'] + 1
          aggregation['H0'].loc['Portfolio']          = df.agg({y: 'mean'}).values
          aggregation['p_val'].loc['Portfolio']       = beta.cdf(aggregation['H0'].loc['Portfolio'], aggregation['alpha'].loc['Portfolio'], aggregation['beta'].loc['Portfolio'])
          return aggregation
          #return:
          #for portfolio and rating classes:
          #1. name of x
          #2. y at the beginning of the relevant observation period
          #3. The number of observations (=N)
          #4. The number of flag=1 (sum of Defaults in case of PD)
          #5. The p-value (one-sided, y > D/N)
     ### Plot ROC
     def ROC_curve(self, df, prediction, default):
          df_train = df[df.training == 1]
          df_valid = df[df.training == 0]
          fpr_train, tpr_train, thresholds = roc_curve(list(df_train[default].values), list(df_train[prediction].values))
          fpr_valid, tpr_valid, thresholds = roc_curve(list(df_valid[default].values), list(df_valid[prediction].values))
          return fpr_train, tpr_train, fpr_valid, tpr_valid
     ###Discrimantory power test: Area Under the Curve###
     def AUC(self, x, y):
          #x : Observed parameter
          #y : Modelled parameter
          #z : Indicate whether VaR calculation is required [True or False]
          ###Calculate parameters###
          N1         = x.sum()
          N2         = len(x)-N1
          R          = y.rank()
          R1         = R.loc[x == True]
          R2         = R.loc[x == False]
          ###Calculate Area Under the Curve###
          U   = N1 * N2 + N1 * (N1 + 1)/2 - R1.sum()
          AUC = U / (N1 * N2)
          #AUC = u.ab.sum()/ (N1 * N2) #Not used (alternative calculation)
          return AUC

# Output validation tests
## Accuracy
acc_ML = CL_PD_tests().Jeffrey(df_ML, 'rating_PD', 'pb_default', 'default')
acc_CL = CL_PD_tests().Jeffrey(df_CL, 'rating_PD', 'pb_default', 'default')
print(acc_ML)
print(acc_CL)
## Distriminatory power
fpr_ML_train, tpr_ML_train, fpr_ML_valid, tpr_ML_valid = CL_PD_tests().ROC_curve(df_ML, 'pb_default', 'default')
fpr_CL_train, tpr_CL_train, fpr_CL_valid, tpr_CL_valid = CL_PD_tests().ROC_curve(df_CL, 'pb_default', 'default')
AUC = np.zeros(6).reshape(3,2)
j   = 0
for i in ['CL', 'ML']:
  AUC[0, j] = CL_PD_tests().AUC(vars()['df_'+i].default,                              vars()['df_'+i].rating_PD)
  AUC[1, j] = CL_PD_tests().AUC(vars()['df_'+i].default[vars()['df_'+i].training==1], vars()['df_'+i].rating_PD[vars()['df_'+i].training==1])
  AUC[2, j] = CL_PD_tests().AUC(vars()['df_'+i].default[vars()['df_'+i].training==0], vars()['df_'+i].rating_PD[vars()['df_'+i].training==0])
  j     += 1
print(AUC)
## Excel
with pd.ExcelWriter(os.getcwd()+'\Validation_output.xlsx') as writer:
  acc_ML.to_excel(writer, sheet_name='Jeffrey ML')
  acc_CL.to_excel(writer, sheet_name='Jeffrey CL')
  pd.DataFrame(AUC).to_excel(writer , sheet_name='AUC')
"""
    Set the plotting stats.
"""
sns.set()
sns.set_context("paper", font_scale=1.5)
sns.set_style("darkgrid", {'font.family': ['EYInterstate']})

"""
     PlotAUC curve
"""

plt.figure(1)
diag_line = np.linspace(0, 1, len(df_CL.default))
plt.plot(diag_line, diag_line, linestyle='--', c='darkgrey')
plt.plot(fpr_ML_valid, tpr_ML_valid, label = "ML (Validation)" + ' : AUC = %0.4f' % auc(fpr_ML_valid,tpr_ML_valid), color='darkblue')
plt.plot(fpr_ML_train, tpr_ML_train, label = "ML (Training)" + ' : AUC = %0.4f' % auc(fpr_ML_train, tpr_ML_train), color='darkblue', ls='--')
plt.plot(fpr_CL_valid, tpr_CL_valid, label = "CL (Validation)" + ' : AUC = %0.4f' % auc(fpr_CL_valid, tpr_CL_valid), color='green')
plt.plot(fpr_CL_train, tpr_CL_train, label = "CL (Training)" + ' : AUC = %0.4f' % auc(fpr_CL_train, tpr_CL_train), color='green', ls='--')
plt.xlabel('True positive rate')
plt.ylabel('False positive rate')
plt.title('Lorenz curve')
plt.legend(loc=4)
plt.show()