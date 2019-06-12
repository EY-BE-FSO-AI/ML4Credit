"""
    Author: Nicolas Bulté
"""

"""
    Import statements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb

from datetime import datetime
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.metrics import roc_curve, r2_score, accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import validation_curve, ShuffleSplit, learning_curve, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from copy import copy
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from functions import plot_learning_curve, plot_multiple_auc, getdummies, plot_correlations, print_dropped_sample_count

    ###################################################################################################################

if __name__ == '__main__':
    plt.close('all')

    """
        Import datasets, select features and define the default-flag collumn.
    """
    
    col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
            'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
            'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
            'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd']
    
    extended_selec_acq = ['LoanID', 'OrLTV', 'LoanPurpose', 'DTIRat', 'PropertyType', 'FTHomeBuyer', 'Channel', 'SellerName','OrInterestRate', 'CreditScore', 'NumBorrow', 'OrDate'] 
    col_acq_subset = extended_selec_acq 
    
    col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
              'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
              'LastInstallDate','ForeclosureDate','DispositionDate','PPRC','AssetRecCost','MHRC',
              'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
              'FPWA','ServicingIndicator'] 
    
    extended_selec_per = ['LoanID', 'MonthsToMaturity', 'CurrInterestRate', 'ForeclosureDate', 'LoanAge', 'CLDS', 'MaturityDate','ZeroBalCode', 'MonthRep']
    
    col_per_subset =  extended_selec_per 
    
    lines_to_read = None
    aquisition_frame = pd.read_csv('Acquisition_2016Q4.txt', sep='|', names=col_acq, usecols=col_acq_subset, index_col=False, nrows=lines_to_read )
    performance_frame = pd.read_csv('Performance_2016Q4.txt', sep='|', names=col_per, usecols=col_per_subset, index_col=False, nrows=lines_to_read) 

    """ Fix the IDs in the observation set by fixing their reporting date AND requiring that the files are healthy. """
    
    observation_frame = performance_frame[(performance_frame.MonthRep == '12/01/2017') & 
                                (   (performance_frame.CLDS == '0') | 
                                    (performance_frame.CLDS == '1') | 
                                    (performance_frame.CLDS == '2')
                                )
                                ]
    obs_ids = observation_frame.LoanID

    """ Load only the observation IDs in the performance frame initially. """
    pf = performance_frame[performance_frame.LoanID.isin(obs_ids)]
    
    """ Keep only the reporting dates that are in our performance period (MM/DD/YYYY format). """
    pf_obs = pf[
                    (pf.MonthRep == '01/01/2018') | 
                    (pf.MonthRep == '02/01/2018') |
                    (pf.MonthRep == '03/01/2018') |
                    (pf.MonthRep == '04/01/2018') |
                    (pf.MonthRep == '05/01/2018') |
                    (pf.MonthRep == '06/01/2018') |
                    (pf.MonthRep == '07/01/2018') |
                    (pf.MonthRep == '08/01/2018') |
                    (pf.MonthRep == '09/01/2018') |
                    (pf.MonthRep == '10/01/2018') |
                    (pf.MonthRep == '11/01/2018') |
                    (pf.MonthRep == '12/01/2018') 
                ]
    
    """ 
    Find the LoanIDs of those loans where a default appears in our performance period.
    """
    pf_obs_defaults = pf_obs[
                            (pf_obs.CLDS != '0') &
                            (pf_obs.CLDS != '1') &
                            (pf_obs.CLDS != '2') &
                            (pf_obs.CLDS != 'X')
                        ].LoanID
    
    pf_obs_defaults.drop_duplicates(keep='last', inplace=True)
    
    """ Merge the acquisition and performance frames. """
    #performance_frame.drop_duplicates(subset='LoanID', keep='last', inplace=True)
    merged_frame = pd.merge(aquisition_frame, observation_frame, on = 'LoanID', how='inner')
    
    merged_frame['Default'] = 0
    merged_frame.loc[merged_frame['LoanID'].isin(pf_obs_defaults), 'Default'] = 1
    
#    performance_frame.drop_duplicates(subset='LoanID', keep='last', inplace=True)
#    
#    performance_frame_obs = performance_frame[(performance_frame.CLDS == '0') | (performance_frame.CLDS == '1') | (performance_frame.CLDS == '2')]
#    print(str(performance_frame_obs.shape[0]) + " healthy samples remaining of initial " + str(performance_frame.shape[0]) + " samples in the performance dataset (after removing duplicates). These samples are removed due to either being in default (CLDS > 2) or with an unknown status (CLDS == X).") 
    
    
    
#    loan_IDs_obs_set = performance_frame_obs.LoanID.values
#
#    performance_frame_1Y_lag = pd.read_csv('Performance_2017Q4.txt', sep='|', names=col_per, usecols=col_per_subset, index_col=False, nrows=lines_to_read) 
#    
#    performance_frame_obs_1Y_lag = performance_frame_1Y_lag[performance_frame_1Y_lag.LoanID.isin(loan_IDs_obs_set)] 


#
#    merged_frame = pd.merge( aquisition_frame, performance_frame, on = 'LoanID', how='inner') 
#    merged_frame.rename(index=str, columns={'ForeclosureDate': 'Default'}, inplace=True)  
#    
#    merged_frame['Default'].fillna(0, inplace=True)
#    merged_frame.loc[merged_frame['Default'] != 0, 'Default'] = 1  
#    merged_frame['Default'] = merged_frame['Default'].astype(int)