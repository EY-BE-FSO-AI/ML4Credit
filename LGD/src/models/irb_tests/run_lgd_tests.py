import numpy as np
import pandas as pd
from lgd_irb_tests import *

if __name__ == '__main__':
    # Test example
    test_lgd = pd.DataFrame({
        'estimated_LGD': np.random.uniform(0, 1, 100),
        'realised_LGD': np.random.uniform(0, 1, 100),
    })
    test_lgd['bin_estimated_LGD'] = pd.qcut(test_lgd['estimated_LGD'], q=10)

    # Create the LGD test object
    lgdtest_obj = LGD_tests()
    # Backtesting
    lgdtest_obj.backtesting(test_lgd, 'estimated_LGD', 'realised_LGD')
    lgdtest_obj.backtesting_facilityGrade(test_lgd, 'estimated_LGD', 'realised_LGD', 'bin_estimated_LGD')
    # Population Stability Index
    lgdtest_obj.psi_lgd(test_lgd, 'estimated_LGD', 'realised_LGD', 'bin_estimated_LGD')


