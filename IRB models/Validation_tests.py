#########################################################################################################################
###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibilty to resolve cartesion product for large lists in AUC function
########################################################################################################################
from scipy.stats import beta
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

class PD_tests(object):
     
     ###Discrimantory power test: Area Under the Curve###
     def AUC(self, x, y, z):
          #x : Observed parameter
          #y : Modelled parameter
          #z : Indicate whether VaR calculation is required [True or False]
          ###Calculate parameters###
          N1         = x.sum()
          N2         = len(x)-N1
          R          = y.rank()
          R1         = R[x == True]
          R2         = R[x == False]
          ###Calculate Area Under the Curve###
          U   = N1 * N2 + N1 * (N1 + 1)/2 - R1.sum()
          AUC = U / (N1 * N2)
          #AUC = u.ab.sum()/ (N1 * N2) #Not used (alternative calculation)
          ###Variance AUC calculation [Optional and can only be applied to small samples]###
          s2 = []
          if z == True:
               u                   = pd.MultiIndex.from_product([R1.tolist(), R2.tolist()]).to_frame()
               u.columns           = ['A', 'B']
               u['ab']             = 0
               u.ab[u.A == u.B]    = 0.5
               u.ab[u.A <  u.B]    = 1
               V10  = u.groupby(['A'], as_index=False)['ab'].sum().iloc[:, 1]/N2/N1
               V01  = u.groupby(['B'], as_index=False)['ab'].sum().iloc[:, 1]/N1/N2
               s2 = np.var(V10, ddof=1) + np.var(V01, ddof=1)
          return AUC, s2
      
     ###Jeffrey's Test

     def Jeffrey(self, name_set):
         
         #name_set is the tested data(sub)set
         #alpha = D + 1/2
         #beta = Nc- D + 1/2
         
         df_Jeffrey = name_set[['grade', 'PD', 'Default_Binary']]
         aggregation = df_Jeffrey.groupby('grade').agg({'grade':'count', 'PD': ['sum','count'], 'Default_Binary': ['sum', 'count']})
         aggregation.loc[len(aggregation) + 1] = aggregation.sum()
         aggregation['actual_DF'] = aggregation[('Default_Binary', 'sum')] / aggregation[('Default_Binary', 'count')]
         aggregation['alpha'] = aggregation[('Default_Binary', 'sum')] + 1/2
         aggregation['beta'] = aggregation[('Default_Binary', 'count')] - aggregation[('Default_Binary', 'sum')] + 1/2
         aggregation['H0PD'] = aggregation[('PD', 'sum')]/aggregation[('PD', 'count')]
         aggregation['p_val'] = beta.cdf(aggregation['H0PD'], aggregation['alpha'], aggregation['beta'])
         aggregation.rename(index={len(aggregation):'Portfolio'})
         return aggregation
    

        #return:
        #for portfolio and rating classes:
        #1. name of the rating grade
        #2. PD at the beginning of the relevant observation period
        #3. The number of customers (=N)
        #4. The number of defaulted customers (=D)
        #5. The p-value (one-sided, PD > D/N)
        #6. (The original exposure at the beginning of the relevant observation period)
    
    ###Concentration in rating grades (2.5.5.3)
    
     def Herfindahl(self, name_set):
        #calculate coefficient of variation
        #calculate HI
        
        #name_set = development_set
        
        df_herf = name_set[['grade']]
        df_herf_agg = df_herf.groupby('grade').agg({'grade':'count'})
        df_herf_agg['R_i'] = df_herf_agg['grade'] / df_herf_agg.grade.sum()
        df_herf_agg['CV_contrib'] = (df_herf_agg['R_i'] - 1 / len(df_herf_agg)) ** 2
        CV = (df_herf_agg['CV_contrib'].sum())**(1/2)
        HI = 1 + math.log((CV**2 + 1) / len(df_herf_agg)) / math.log(len(df_herf_agg))
        CV_p_val = "placeholder" #CV initial period to be computed and both are to be compared.
        
        return CV, HI, CV_p_val
