#########################################################################################################################
###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibilty to resolve cartesion product for large lists in AUC function
########################################################################################################################
from scipy.stats import beta
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import t
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
        CV = (len(df_herf_agg) * df_herf_agg['CV_contrib'].sum())**(1/2)
        HI = 1 + math.log((CV**2 + 1) / len(df_herf_agg)) / math.log(len(df_herf_agg))
        CV_p_val = "placeholder" #CV initial period to be computed and both are to be compared.
        
        return CV, HI, CV_p_val

     def MWB(self, abs_freq, rel_freq):
        """
        Compute the matrix weighted bandwidth metric (§ 2.5.5.1);
        :param abs_freq: transition matrix(KxK)with number of customer changes per grades;
        :param rel_freq: transition matrix(KxK) with relative frequency of customer grade change.
        :return: upper_MWB: upper matrix bandwidth metric, lower_MWB: lower matrix bandwidth metric.
        """""
         
        abs_freq = abs_freq.as_matrix()
        rel_freq = rel_freq.as_matrix()

        n_i = abs_freq.sum(axis = 1)
        K = len(abs_freq)
         
        M_norm_u = 0
        for i in range(1, K - 1):
            M_scalar_i = 0
            sum_rel_frq_row = 0
            for j in range(i+1, K):
                M_scalar_i = max(abs(i - K), abs(i-1))
                sum_rel_frq_row = sum_rel_frq_row + rel_freq[i,j]
        
            M_norm_u += M_scalar_i * sum_rel_frq_row * n_i[i]
        
        M_norm_l = 0
        for i in range(2, K):
            M_scalar_i = 0
            sum_rel_frq_row = 0
            for j in range(1, i - 1):
                M_scalar_i = max(abs(i - K), abs(i-1))
                sum_rel_frq_row = sum_rel_frq_row + rel_freq[i,j]
            
            M_norm_l += M_scalar_i * sum_rel_frq_row * n_i[i]
        
        temp = 0
        for i in range(1, K-1):
            for j in range(i + 1, K):
                temp += np.abs(i - j) * n_i[i] * rel_freq[i, j]
        upper_MWB = (1 / M_norm_u) * temp 
        
        temp = 0
        for i in range(2, K):
            for j in range(1, i - 1):
                temp += np.abs(i - j) * n_i[i] * rel_freq[i, j]
        lower_MWB = (1 / M_norm_l) * temp
        
        
        return upper_MWB, lower_MWB

     def stability_migration_matrix(self, abs_freq, rel_freq):
         """
         Compute the migration stability matrix metrics (§ 2.5.5.2);
         To be reported to the regulator: the relative frequencies of transitions between rating grades, values
         for the test statistic z_ij and associated p-values.
         :param abs_freq: transition matrix (KxK) with number of customer changes per grades;
         :param rel_freq: transition matrix (KxK) with relative frequency of customer grade change.
         :return: z_up, z_low, zUP_pval, zDOWN_pval
         """
         N = abs_freq.sum(axis=1).values
         p = rel_freq.as_matrix()
         K = len(p)

         z_up = np.zeros(p.shape)
         for i in range(1, K - 1):
             for j in range(i + 1, K):
                 up = p[i, j - 1] - p[i, j]
                 down = np.sqrt(
                 (p[i, j] * (1 - p[i, j]) + p[i, j - 1] * (1 - p[i, j - 1]) + 2 * p[i, j] * p[i, j - 1]) / N[i])
                 z_up[i, j] = up / down

         z_low = np.zeros(p.shape)
         for i in range(1, K):
             for j in range(0, i):
                 up = p[i, j + 1] - p[i, j]
                 down = np.sqrt(
                 (p[i, j] * (1 - p[i, j]) + p[i, j + 1] * (1 - p[i, j + 1]) + 2 * p[i, j] * p[i, j + 1]) / N[i])
                 z_low[i, j] = up / down
         zUP_pval= norm.sf(abs(z_up)) # one-sided
         zDOWN_pval = norm.sf(abs(z_low)) # one-sided

         return z_up, z_low, zUP_pval, zDOWN_pval

class LGD_tests(object):

    def backtesting(self, data_set):
        """
        NOT THE FINAL VERSION
        LGD Back-testing using a t-test.
        Null-hypothesis is: estimatedLGD > realisedLGD. The test statistic is asymptotically Student-t distributed with
        N-1 degrees of freedom.
        :param data_set: development/monitoring sets
        :return:
        """
        #### Select estimated and realised LGDs for defaulting facilities
        estimatedLGD = data_set[data_set.Default_Binary == 1].LGD_predicted
        realisedLGD = data_set[data_set.Default_Binary == 1].LGD_realised
        N = len(data_set[data_set.Default_Binary == 1])

        ### Construct test statistic 2.6.2.1
        s2 = (((realisedLGD - estimatedLGD).sum() - (realisedLGD - estimatedLGD).mean() )**2 ) / (N - 1)
        t_stat = np.sqrt(N) * ((realisedLGD - estimatedLGD).mean()) / np.sqrt( s2 )

        pval = 1 - t.cdf(t_stat, df=N-1) #one-sided

        return pval
    
class CCF_tests(object):
    
    def backtesting(self, data_set):
        
        #### Select estimated and realised LGDs for defaulting facilities
        estimatedCCF = data_set.CCF_predicted
        realisedCCF = data_set.CCF_realised
        R = len(data_set)
        
        ### Construct test statistic 2.9.3.1
        s2 = (((realisedCCF - estimatedCCF).sum() - (realisedCCF - estimatedCCF).mean() )**2 ) / (R - 1)
        t_stat = np.sqrt(R) * ((realisedCCF - estimatedCCF).mean()) / np.sqrt( s2 )

        CCF_pval = 1 - t.cdf(t_stat, df=R-1) #one-sided
        
        return CCF_pval
    
    def gAUC(self, data_set):
        #predicted = predicted CCFs
        #realized = realized CCFs
        
        
        return gAUC, s2