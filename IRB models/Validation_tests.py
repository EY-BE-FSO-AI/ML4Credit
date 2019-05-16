#########################################################################################################################
###Owner code: Brent Oeyen
###Comments: 
###          -Look into the possibilty to resolve cartesion product for large lists in AUC function
########################################################################################################################
from scipy.stats          import beta
from scipy.stats          import norm
from scipy.stats          import binom
from scipy.stats          import t
from dask.multiprocessing import get
import matplotlib.pyplot as plt
import numpy             as np
import pandas            as pd
import _thread           as th
import dask.dataframe    as dd
import math
import itertools
import random

class general(object):

     def test_stat(self, x, y, z):
          out = (x-y)/z
          return out

     def p_value(self, x):
          out = 1 - norm.cdf(x)
          return out

class matrix(object):
     
     def check_square(self, M):
          #Check whether matrix MemoryError is square
          result = True if M.shape[0] == M.shape[1] else False
          return result
     
     def square(self, M):
          #Make matrix square in case it is not
          if self.check_square(M) == False:
               r   = M.shape[0] #Number of rows
               c   = M.shape[1] #Number of columns
               m   = max(r, c)  #Maximum between rows and columns
               for i in M.index:
                   if i not in M.columns:
                       M[i] = [0] * len(M.index)
               for j in M.columns:
                   if j not in M.index:
                       M.loc[j, :] = [0] * len(M.columns)
               M = M.sort_index(axis=0).sort_index(axis=1)
               # for i in range(1,(m+1)):
               #      if i not in M.index:
               #           row            = pd.DataFrame([[0] *c])
               #           row.index      = [i]
               #           row.columns    = range(1, (c+1))
               #           M              = M.append(row)
               # M = M.sort_index()
               # for i in range(1,(m+1)):
               #      if i not in M.columns:
               #           M.insert(i-1, i, 0)
          return M
     
     ###Calculate observations for a 2D matrix###
     def matrix_obs(self, data, x, y, z):
          #x: variable used to create the rows of the matrix
          #y: variable used to create the columns of the matrix
          #z: variable used to calculate the sum per cell of the matrix
          matrix  = data[data[z] == 0].groupby([x, y]).size().unstack(fill_value=0)
          matrix  = self.square(matrix)
          return matrix
          
     def matrix_prob(self, matrix):
          #matrix: the returned value of the matrix_obs function
          matrix_ = matrix / matrix.sum(axis=0)
          return matrix_.fillna(0)

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
          R1         = R.loc[x == True]
          R2         = R.loc[x == False]
          ###Calculate Area Under the Curve###
          U   = N1 * N2 + N1 * (N1 + 1)/2 - R1.sum()
          AUC = U / (N1 * N2)
          #AUC = u.ab.sum()/ (N1 * N2) #Not used (alternative calculation)
          ###Variance AUC calculation [Optional and can only be applied to small samples]###
          s2 = []
          if z == True:
               def aggregate(t1, t2): return t1.apply(lambda a: sum((a==t2)*0.5 + (a<t2)*1))
               Ua        = dd.map_partitions(aggregate, dd.from_pandas(R1, npartitions=4), R2).compute(get=get)
               Ub        = dd.map_partitions(aggregate, dd.from_pandas(R2, npartitions=4), R1).compute(get=get)
               V10       = Ua/N2/N1
               V01       = Ub/N1/N2
               s2        = np.var(V10, ddof=1) + np.var(V01, ddof=1)
          return AUC, s2
      
     ###Jeffrey's Test

     def Jeffrey(self, name_set):
         
         #name_set is the tested data(sub)set
         #alpha = D + 1/2
         #beta = Nc- D + 1/2
         
         df_Jeffrey                               = name_set[['grade', 'PD', 'Default_Binary']]
         df_Jeffrey[['PD', 'Default_Binary']]     = name_set[['PD', 'Default_Binary']].apply(lambda x: pd.to_numeric(x, errors='coerce'))
         aggregation                              = df_Jeffrey.groupby('grade').agg({'grade':'count', 'PD': ['sum', 'count', 'mean'], 'Default_Binary': ['sum', 'count', 'mean']})
         aggregation.loc[len(aggregation) + 1]    = aggregation.sum()
         aggregation['actual_DF']                 = aggregation[('Default_Binary', 'mean')]
         aggregation['alpha']                     = aggregation[('Default_Binary', 'sum')] + 1/2
         aggregation['beta']                      = aggregation[('Default_Binary', 'count')] - aggregation[('Default_Binary', 'sum')] + 1/2
         aggregation['H0PD']                      = aggregation[('PD', 'mean')]
         aggregation['p_val']                     = beta.cdf(aggregation['H0PD'], aggregation['alpha'], aggregation['beta'])
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

     def Herfindahl(self, development, validation):
         """
         Calculate Coefficient of Variation and Herfindahl Index for initial and current period,
         and associated p value (see §2.5.5.3).
         :param development:
         :param validation:
         :return:
         """
         K = len(development[development.Default_Binary == 0].grade.unique())
         CV_init, HI_init = self.calculate_Herfindahl( development )
         CV_curr, HI_curr = self.calculate_Herfindahl( validation )
         cr_pval = 1 - norm.cdf(np.sqrt(K - 1) * (CV_curr - CV_init) / np.sqrt(CV_curr ** 2 * (0.5 + CV_curr ** 2)))
         return CV_init, HI_init, CV_curr, HI_curr, cr_pval
    
     def calculate_Herfindahl(self, name_set):
        #calculate coefficient of variation
        #calculate HI
        
        #name_set = development_set
        
        df_herf = name_set[['grade']]
        df_herf_agg = df_herf.groupby('grade').agg({'grade':'count'})
        df_herf_agg['R_i'] = df_herf_agg['grade'] / df_herf_agg.grade.sum()
        df_herf_agg['CV_contrib'] = (df_herf_agg['R_i'] - 1 / len(df_herf_agg)) ** 2
        CV = (len(df_herf_agg) * df_herf_agg['CV_contrib'].sum())**(1/2)
        HI = 1 + math.log((CV**2 + 1) / len(df_herf_agg)) / math.log(len(df_herf_agg))
        return CV, HI

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
         for i in range(0, K - 1):
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
         zUP_pval= norm.cdf(z_up) # one-sided
         zDOWN_pval = norm.cdf(z_low) # one-sided

         return z_up, z_low, zUP_pval, zDOWN_pval

class LGD_tests(object):

    def backtesting(self, data_set, x, y):
        """
        LGD Back-testing using a t-test.
        Null-hypothesis is: estimatedLGD > realisedLGD. The test statistic is asymptotically Student-t distributed with
        N-1 degrees of freedom.
        :param data_set: development/monitoring sets
        :param x:
        :param y:
        :return:
        """
        #### Select estimated and realised LGDs for defaulting facilities
        estimatedLGD = data_set[data_set.Default_Binary == 1][x]
        realisedLGD = data_set[data_set.Default_Binary == 1][y]
        N = len(data_set[data_set.Default_Binary == 1])

        ### Construct test statistic 2.6.2.1
        s2 = (((realisedLGD - estimatedLGD).sum() - (realisedLGD - estimatedLGD).mean() )**2 ) / (N - 1)
        t_stat = np.sqrt(N) * ((realisedLGD - estimatedLGD).mean()) / np.sqrt( s2 )

        pval = 1 - t.cdf(t_stat, df=N-1) #one-sided

        return [t_stat, s2, pval]

    def backtesting_facilityGrade(self, data_set, x, y):
        """
        Similar to backtesting function. Returns backtesting results per facility grades.
        :param data_set:
        :param x:
        :param y:
        :return: backtesting results per facility grade -> test statistic, LGD estimated variance and p-value.
        """
        res = {}
        for bin in data_set["Bin_{s}".format(s=x)].unique():
            data_set_perGrade = data_set[data_set["Bin_{s}".format(s=x)] == bin]
            res[bin] = self.backtesting(data_set_perGrade, x, y)
        return res

    def gAUC_LGD(self, current_transMatrix, initial_transMatrix):
        """
        Computes the generalised AUC and test statistic for LGD (§2.6.3.1).
        :param current_transMatrix: Pandas dataframe (LxL) with L grades/clusters for current observation period;
        :param initial_transMatrix: Pandas dataframe (LxL) with L grades/clusters for initial observation period;
        :return: Initial gAUC, current gAUC, test stat and p value.
        """
        gAUC_init, s_init = gAUC(initial_transMatrix).compute_gAUC_s()
        gAUC_curr, s_curr = gAUC(current_transMatrix).compute_gAUC_s()

        S      = general().test_stat(gAUC_init, gAUC_curr, s_curr)
        p_val  = general().p_value(S)

        return gAUC_init, gAUC_curr, S, s_curr, s_init, p_val

    def psi_lgd(self, data_set):
          ### Population stability index
          grade = data_set.groupby('grade').agg({'LGD_realised': 'mean', 'LGD': 'mean'})
          PSI = 0
          for i in range(0, len(grade)):
               PSI += (grade.iloc[i, 1] - grade.iloc[i, 0]) * np.log(grade.iloc[i, 1] / grade.iloc[i, 0])
          return PSI
    
class CCF_tests(object):

    def backtesting(self, data_set, x, y):
        """
        CCF Back-testing using a t-test.
        Null-hypothesis is: estimatedCCF > realisedCCF. The test statistic is asymptotically Student-t distributed with
        N-1 degrees of freedom.
        :param data_set: development/monitoring sets
        :param x:
        :param y:
        :return:
        """
        #### Select estimated and realised LGDs for defaulting facilities
        estimatedCCF = data_set[data_set.Default_Binary == 1][x]
        realisedCCF = data_set[data_set.Default_Binary == 1][y]
        N = len(data_set[data_set.Default_Binary == 1])

        ### Construct test statistic 2.6.2.1
        s2 = (((realisedCCF - estimatedCCF).sum() - (realisedCCF - estimatedCCF).mean() )**2 ) / (N - 1)
        t_stat = np.sqrt(N) * ((realisedCCF - estimatedCCF).mean()) / np.sqrt( s2 )

        pval = 1 - t.cdf(t_stat, df=N-1) #one-sided

        return [t_stat, s2, pval]

    def backtesting_facilityGrade(self, data_set, x, y):
        """
        Similar to backtesting function. Returns backtesting results per facility grades.
        :param data_set:
        :return: backtesting results per facility grade -> test statistic, LGD estimated variance and p-value.
        """
        res = {}
        for bin in data_set["Bin_{s}".format(s=x)].unique():
            data_set_perGrade = data_set[data_set["Bin_{s}".format(s=x)] == bin]
            res[bin] = self.backtesting(data_set_perGrade, x, y)
        return res

    def gAUC_CCF(self, current_transMatrix, initial_transMatrix):
        """
        Computes the generalised AUC and test statistic for CCF (§2.9.4.1).
        :param current_transMatrix: Pandas dataframe (LxL) with L grades/clusters for current observation period;
        :param initial_transMatrix: Pandas dataframe (LxL) with L grades/clusters for initial observation period;
        :return: Initial gAUC, current gAUC, test stat and p value.
        """
        gAUC_init, s_init = gAUC(initial_transMatrix).compute_gAUC_s()
        gAUC_curr, s_curr = gAUC(current_transMatrix).compute_gAUC_s()
        S         = general().test_stat(gAUC_init, gAUC_curr, s_curr)
        p_val     = general().p_value(S)
        return gAUC_init, gAUC_curr, S, s_curr, s_init, p_val

    def psi_ccf(self, data_set):
        ### Population stability index
        grade = data_set.groupby('grade').agg({'CCF': 'mean', 'CCF_': 'mean'})
        PSI = 0
        for i in range(0, len(grade)):
            PSI += (grade.iloc[i, 1] - grade.iloc[i, 0]) * np.log(grade.iloc[i, 1] / grade.iloc[i, 0])
        return PSI



class gAUC(object):
    """
    Class object for generalized AUC calculation. See IRB documentation annex 3.2.
    """

    def __init__(self, transMatrix):
        """
        :param transMatrix: transition matrix
        """
        self.transMatrix = transMatrix

    def A_lower_ij(self, trans_matrix, i, j):
        temp = 0
        for k in range (0, i):
            for l in range(0,j):
                temp += trans_matrix.iloc[k,l]
        return temp

    def A_higher_ij(self, trans_matrix, i, j):
        temp = 0
        for k in range (i, len(trans_matrix)):
            for l in range(j, len(trans_matrix)):
                temp += trans_matrix.iloc[k,l]
        return temp

    def D_left_ij(self, trans_matrix, i, j):
        temp = 0
        for k in range (i, len(trans_matrix)):
            for l in range(0,j):
                temp += trans_matrix.iloc[k,l]
        return temp

    def D_right_ij(self, trans_matrix, i, j):
        temp = 0
        for k in range (0, i):
            for l in range(j, len(trans_matrix)):
                temp += trans_matrix.iloc[k,l]
        return temp

    def compute_gAUC_s(self):
        """
        Compute the gAUC and its standard deviation (see annex 3.2).
        :param transition_matrix: as it says.
        :return:
        """
        transition_matrix_A  = np.zeros((len(self.transMatrix), len(self.transMatrix)))
        transition_matrix_D  = np.zeros((len(self.transMatrix), len(self.transMatrix)))
        d                    = np.zeros((len(self.transMatrix), len(self.transMatrix)))
        P                    = 0
        Q                    = 0
        for i in range(0, len(self.transMatrix)):
            for j in range(0, len(self.transMatrix)):
                if i == j:
                    transition_matrix_A[i, j]    =  0
                    transition_matrix_D[i, j]    =  0
                    P                            += self.transMatrix.iloc[i, j] * transition_matrix_A[i, j]
                    Q                            += self.transMatrix.iloc[i, j] * transition_matrix_D[i, j]
                    d[i, j]                      =  transition_matrix_A[i, j] - transition_matrix_D[i, j]
                else:
                    transition_matrix_A[i, j]    =  self.A_lower_ij(self.transMatrix, i, j) \
                                                    + self.A_higher_ij(self.transMatrix, i, j)
                    transition_matrix_D[i, j]    =  self.D_left_ij(self.transMatrix, i, j) \
                                                    + self.D_right_ij(self.transMatrix, i, j)
                    P                            += self.transMatrix.iloc[i, j] * transition_matrix_A[i, j]
                    Q                            += self.transMatrix.iloc[i, j] * transition_matrix_D[i, j]
                    d[i, j]                      =  transition_matrix_A[i, j] - transition_matrix_D[i, j]

        r          = self.transMatrix.sum(axis=0).values
        F          = self.transMatrix.values.sum() ** 2
        w_r        = F - (np.sum(r ** 2))
        somersD    = (P - Q) / w_r
        gAUC       = (somersD + 1) / 2

        # gAUC's standard deviation estimation:
        s_rhs = 0
        for i in range(0, len(self.transMatrix)):
            for j in range(0, len(self.transMatrix)):
                s_rhs += self.transMatrix.iloc[i, j] * (w_r * d[i, j] - (P - Q) * (F - r[i])) ** 2

        s = (1 / w_r ** 2) * np.sqrt(s_rhs)

        return gAUC, s

class extra_tests:
     
     def range(self, data, code):
          ###Calculate metric for each observation year in the development data set
          out  = []
          for i in data.obs_dt.dt.year.unique():
               df   = data[data.obs_dt.dt.year == i]
               exec(code)
          return out
     
     def boot(self, data, code, n, size):
          ###Boostrap empirical distribution of a metric in the development data set
          #n: Number of samples
          #size: Sample size
          out            = []
          random.seed    = 1
          for i in range(n):
               sample = data.iloc[random.sample(range(data.shape[0]-1), size), :]
               exec(code)
          return out