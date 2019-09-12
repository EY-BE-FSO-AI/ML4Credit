"""
IRB Reporting - LGD tests
"""
from scipy.stats          import beta, norm, binom, t
import numpy                  as np
import pandas                 as pd
import math
import random

class general(object):

    def test_stat(self, x, y, z):
        out = (x - y) / z
        return out

    def p_value(self, x):
        out = 1 - norm.cdf(x)
        return out

class LGD_tests(object):

    def backtesting(self, data_set, x, y):
        """
        LGD Back-testing using a t-test.
        Null-hypothesis is: estimatedLGD > realisedLGD. The test statistic is asymptotically Student-t distributed with
        N-1 degrees of freedom.
        :param data_set: pd.DataFrame - development/monitoring sets
        :param x: Str - Label of estimated LGD
        :param y: Str - Label of realised LGD
        :return: list - backtesting results for portfolio -> test statistic, LGD estimated variance and p-value.
        """
        #### Select estimated and realised LGDs for defaulting facilities
        estimatedLGD = data_set[x] #data_set[data_set.Default_Binary == 1][x]
        realisedLGD = data_set[y] #data_set[data_set.Default_Binary == 1][y]
        N = len(data_set) #len(data_set[data_set.Default_Binary == 1])

        ### Construct test statistic 2.6.2.1
        s2 = (((realisedLGD - estimatedLGD) - (realisedLGD - estimatedLGD).mean() )**2 ).sum() / (N - 1)
        t_stat = np.sqrt(N) * ((realisedLGD - estimatedLGD).sum()) / np.sqrt( s2 )

        pval = 1 - t.cdf(t_stat, df=N-1) #one-sided

        return [s2, t_stat, pval]

    def backtesting_facilityGrade(self, data_set, x, y, bin_label):
        """
        Similar to backtesting function. Returns backtesting results per facility grades.
        :param data_set: pd.DataFrame - development/monitoring sets
        :param x: Str - Label of estimated LGD in data_set
        :param y: Str - Label of realised LGD in data_set
        :param bin_label: Str - Label of bins/segments in data_set
        :return: list - backtesting results per facility grade -> test statistic, LGD estimated variance and p-value.
        """
        res = {}
        for bin in sorted(data_set[bin_label].unique()):
            data_set_perGrade = data_set[data_set[bin_label] == bin]
            res[bin] = self.backtesting(data_set_perGrade, x, y)
        return res

    def gAUC_LGD(self, current_transMatrix, initial_transMatrix):
        """
        Computes the generalised AUC and test statistic for LGD (§2.6.3.1).
        :param current_transMatrix: pd.dataframe - matrix (LxL) with L grades/clusters for current observation period;
        :param initial_transMatrix: pd.dataframe - matrix (LxL) with L grades/clusters for initial observation period;
        :return: Initial gAUC, current gAUC, test stat, current var, initial var and p value.
        """
        gAUC_init, s_init = gAUC(initial_transMatrix).compute_gAUC_s()
        gAUC_curr, s_curr = gAUC(current_transMatrix).compute_gAUC_s()

        S      = general().test_stat(gAUC_init, gAUC_curr, s_curr)
        p_val  = general().p_value(S)

        return gAUC_init, gAUC_curr, S, s_curr, s_init, p_val

    def psi_lgd(self, data_set, x, y, bin_label):
        """
        Population stability index
        :param data_set: pd.DataFrame - development/monitoring sets
        :param x: Str - Label of estimated LGD in data_set
        :param y: Str - Label of realised LGD in data_set
        :param bin_label: Str - Label of bins/segments in data_set
        :return: 
        """
        grade = data_set.groupby( bin_label ).agg({ y : 'mean', x : 'mean'})
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
        :param transition_matrix: pd.DataFrame - development/monitoring transition matrix
        :return: gAUC and variance
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


class matrix(object):

    def check_square(self, M):
        # Check whether matrix MemoryError is square
        result = True if M.shape[0] == M.shape[1] else False
        return result

    def square(self, M):
        # Make matrix square in case it is not
        if self.check_square(M) == False:
            r = M.shape[0]  # Number of rows
            c = M.shape[1]  # Number of columns
            m = max(r, c)  # Maximum between rows and columns
            for i in M.index:
                if i not in M.columns:
                    M[i] = [0] * len(M.index)
            for j in M.columns:
                if j not in M.index:
                    M.loc[j, :] = [0] * len(M.columns)
            M = M.sort_index(axis=0).sort_index(axis=1)
        return M

    ###Calculate observations for a 2D matrix###
    def matrix_obs(self, data, x, y, z):
        # x: variable used to create the rows of the matrix
        # y: variable used to create the columns of the matrix
        # z: variable used to calculate the sum per cell of the matrix
        if y == 'Bin_PD':
            matrix = data[data[z] == 0].groupby([x, y]).size().unstack(fill_value=0)
            matrix["dflt"] = data[data[z] == 1].groupby([x]).size()
            matrix["hiii"] = len(matrix.index) * [0]  # Placeholder for pt h.iii in 2.5.1
            matrix["hiv"] = len(matrix.index) * [0]  # Placeholder for pt h.iv in 2.5.1
            if matrix.sum(axis=1).sum() != len(data):
                raise ValueError("Transition Matrix customer number different from total customer number.")
        else:
            matrix = data.groupby([x, y]).size().unstack(fill_value=0)
            matrix = self.square(matrix)
        return matrix

    def matrix_prob(self, matrix):
        # matrix: the returned value of the matrix_obs function
        matrix_ = matrix.div(matrix.sum(axis=1), axis=0)
        return matrix_.fillna(0)

def cut(array, bins):
    labels = ['({b0}, {b1}]'.format(b0=b[0], b1=b[1]) for b in bins]
    intervals = [pd.Interval(*b) for b in bins]

    categories = []
    for value in array:
        cat = None
        for i, interval in enumerate(intervals):
            if value in interval:
                cat = labels[i]
                break
        categories.append(cat)

    return categories
