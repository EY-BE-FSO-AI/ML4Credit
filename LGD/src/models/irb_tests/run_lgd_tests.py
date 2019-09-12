import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from irb_tests.lgd_irb_tests import LGD_tests, gAUC, matrix, cut

if __name__ == '__main__':
    # Test example: continious LGD
    test_lgd = pd.DataFrame({
        'estimated_LGD': np.random.beta(1/2, 1/2, 2000),
        'realised_LGD': np.random.beta(1, 3, 2000),
    })

    # Transition matrix
    segments = [(0, 0.05, 'both'), (0.05, 0.1, 'right'), (0.1, 0.2, 'right'), (0.2, 0.3, 'right'), (0.3, 0.4, 'right'),
                (0.4, 0.5, 'right'), (0.5, 0.6, 'right'), (0.6, 0.7, 'right'), (0.7, 0.8, 'right'), (0.8, 0.9, 'right'),
                (0.9, 1, 'neither'), (1, 1.1, 'left')]  # IRB recommands these segments for cont LGD
    test_lgd['bin_estimated_LGD'] = cut(test_lgd['estimated_LGD'], segments)
    test_lgd['bin_realised_LGD'] = cut(test_lgd['realised_LGD'], segments)
    tranisition_matrix = matrix().matrix_obs(data=test_lgd, x='bin_estimated_LGD', y='bin_realised_LGD', z=None)
    tranisition_matrix_freq = matrix().matrix_prob( tranisition_matrix )

    # Create the LGD test object
    lgdtest_obj = LGD_tests()
    # Backtesting
    var, t_stat, p_val = lgdtest_obj.backtesting(test_lgd, 'estimated_LGD', 'realised_LGD')
    print('Backtesting -> Variance: {v}; T-statistic: {t}; P-value: {p}'.format(v=var, t=t_stat, p=p_val))
    backtesting_per_bin = lgdtest_obj.backtesting_facilityGrade(test_lgd, 'estimated_LGD', 'realised_LGD', 'bin_estimated_LGD')
    # Population Stability Index
    psi = lgdtest_obj.psi_lgd(test_lgd, 'estimated_LGD', 'realised_LGD', 'bin_estimated_LGD')
    print('Population stability index: {p}'.format(p=psi))
    # Generalized AUC
    gauc_obj = gAUC(transMatrix= tranisition_matrix_freq)
    print( 'Generalised AUC (gAUC, variance): ', gauc_obj.compute_gAUC_s() )
