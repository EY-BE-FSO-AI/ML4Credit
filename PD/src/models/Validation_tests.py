import os
import pandas                      as pd
import numpy                       as np
from scipy.stats          import beta
from scipy.stats          import norm
from scipy.stats          import binom
from scipy.stats          import t

# Import ML output
df_ML                = pd.read_csv(os.getcwd()+'\predictions_train.csv').iloc[:, 1:]
df_ML['training']    = 1
df_dummy             = pd.read_csv(os.getcwd()+'\predictions_validation.csv').iloc[:, 1:]
df_dummy['training'] = 0
df_ML                = df_ML.append(df_dummy).sort_values(by='LoanID').reset_index().iloc[:, 1:]
df_ML['default']     = np.random.uniform()     #TO BE REPLACED BY THE ACTUAL DEFAULT FLAG

# Master Scale
ratings            = 22                                                                           #Number of rating grades
PD_min             = 0.0001                                                                       #Minimum PD value (regulatory threshold)
slope              = (np.log((2-PD_min)/PD_min)-0)/(ratings-1)                                    #Slope between min PD value and default in logit space, depending on NB of rating grades
MS                 = 2/(1+np.exp(slope*pd.Series(list(range(ratings)))))
idx                = pd.IntervalIndex.from_arrays(MS[1:].append(pd.Series(0)), MS, closed='left')
df_ML['rating_PD'] = MS[idx.get_indexer(df_ML.pb_default)].values

# Validation test - Accuracy
class PD_tests(object):      
     ###Jeffrey's Test
     def Jeffrey(self, df, x, y , z):
          #df is a dataframe object that contains columns x, y and z
          #x aggregation variable (rating grade for PD test)
          #y modelled variable
          #z observed variable (expected to be binary)
          #alpha = D + 1/2
          #beta = Nc- D + 1/2
          aggregation                              = df.groupby(x).agg({x:'count', y: ['sum', 'count', 'mean'], z: ['sum', 'count', 'mean']})
          aggregation['Observed']                  = aggregation[(z, 'mean')]
          aggregation['alpha']                     = aggregation[(z, 'sum')] + 1/2
          aggregation['beta']                      = aggregation[(z, 'count')] - aggregation['alpha'] + 1
          aggregation['H0']                        = aggregation[(y, 'mean')]
          aggregation['p_val']                     = beta.cdf(aggregation['H0'], aggregation['alpha'], aggregation['beta'])
          aggregation.loc['Portfolio']             = aggregation.sum()
          aggregation['Observed'].loc['Portfolio'] = df.agg({z: 'mean'}).values
          aggregation['alpha'].loc['Portfolio']    = df.agg({z: 'sum' }).values + 1/2
          aggregation['beta'].loc['Portfolio']     = df.agg({z: 'count'}).values - aggregation['alpha'].loc['Portfolio'] + 1
          aggregation['H0'].loc['Portfolio']       = df.agg({y: 'mean'}).values
          aggregation['p_val'].loc['Portfolio']    = beta.cdf(aggregation['H0'].loc['Portfolio'], aggregation['alpha'].loc['Portfolio'], aggregation['beta'].loc['Portfolio'])
          return aggregation
          #return:
          #for portfolio and rating classes:
          #1. name of x
          #2. y at the beginning of the relevant observation period
          #3. The number of observations (=N)
          #4. The number of flag=1 (sum of Defaults in case of PD)
          #5. The p-value (one-sided, y > D/N)

acc = PD_tests().Jeffrey(df_ML, 'rating_PD', 'pb_default', 'default')
print(acc)