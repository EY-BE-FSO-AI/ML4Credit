"""
    Author: Nicolas Bulté
"""
"""
Glossary mapping
"""

# LoanID               = Loan Identifier (A,P)
# MonthRep             = Monthly Reporting Period (P)
# Servicer             = Servicer Name (P)
# CurrInterestRate     = CURRENT INTEREST RATE (P)
# CAUPB                = CURRENT ACTUAL UNPAID PRINCIPAL BALANCE (P)
# LoanAge              = Loan Age (P)
# MonthsToMaturity     = Remaining Months to Legal Maturity (P)
# AdMonthsToMaturity   = ADJUSTED REMAINING MONTHS TO MATURITY (P)
# MaturityDate         = Maturity Date (P)
# MSA                  = Metropolitan Statistical Area (P)
# CLDS                 = Current Loan Delinquency Status (P)
# ModFlag              = Modification Flag (P)
# ZeroBalCode          = Zero Balance Code (P)
# ZeroBalDate          = Zero Balance Effective Date(P)
# LastInstallDate      = LAST PAID INSTALLMENT DATE
# ForeclosureDate      = FORECLOSURE DATE
# DispositionDate      = DISPOSITION DATE
# ForeclosureCosts     = FORECLOSURE COSTS (P)
# PPRC                 = Property Preservation and Repair Costs (P)
# AssetRecCost         = ASSET RECOVERY COSTS (P)
# MHEC                 = Miscellaneous Holding Expenses and Credits (P)
# ATFHP                = Associated Taxes for Holding Property (P)
# NetSaleProceeds      = Net Sale Proceeds (P)
# CreditEnhProceeds    = Credit Enhancement Proceeds (P)
# RPMWP                = Repurchase Make Whole Proceeds(P)
# OFP                  = Other Foreclosure Proceeds (P)
# NIBUPB               = Non-Interest Bearing UPB (P)
# PFUPB                = PRINCIPAL FORGIVENESS UPB (P)
# RMWPF                = Repurchase Make Whole Proceeds Flag (P)
# FPWA                 = Foreclosure Principal Write-off Amount (P)
# ServicingIndicator   = SERVICING ACTIVITY INDICATOR (P)

"""
    Import statements 
"""

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import datasets, select features and define the default-flag collumn.
col_per = ['LoanID', 'MonthRep', 'Servicer', 'CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity',
           'AdMonthsToMaturity', 'MaturityDate', 'MSA', 'CLDS', 'ModFlag', 'ZeroBalCode', 'ZeroBalDate',
           'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'ForeclosureCosts', 'PPRC', 'AssetRecCost', 'MHEC',
           'ATFHP', 'NetSaleProceeds', 'CreditEnhProceeds', 'RPMWP', 'OFP', 'NIBUPB', 'PFUPB', 'RMWPF',
           'FPWA', 'ServicingIndicator']

# Python will guess the datatypes not specified in the map function, for dates the dtype will be 'object'. (hence: here all dates)
# If an expected integer variables contains NaN values it will be set to 'float32'
perf_type_map = {'LoanID': 'int64', 'Servicer': 'category', 'CurrInterestRate': 'float32', 'CAUPB': 'float32',
                 'LoanAge': 'int64', 'MonthsToMaturity': 'int64', 'AdMonthsToMaturity': 'float32', 'MSA': 'category',
                 'CLDS': 'category', 'ModFlag': 'category', 'ZeroBalCode': 'float32', 'ForeclosureCosts': 'float32',
                 'PPRC': 'float32', 'AssetRecCost': 'float32', 'MHEC': 'float32', 'ATFHP': 'float32',
                 'NetSaleProceeds': 'float32', 'CreditEnhProceeds': 'float32', 'RPMWP': 'float32', 'OFP': 'float32',
                 'NIBUPB': 'float32', 'PFUPB': 'float32', 'RMWPF': 'category', 'FPWA': 'float32',
                 'ServicingIndicator': 'category'}

extended_selec_per = ['LoanID', 'MonthRep', 'CLDS']

col_per_subset = extended_selec_per


def read_file(file_name, ref_year, lines_to_read=None):
    """
    Read file in function to avoid memory issues
    + Add lagged payment variables
    Parameters
    ----------
    file_name: Path name of the file;
    ref_year: Specify the list of years to be read, if None-> whole dataset is used;
    lines_to_read: Specify the number of rows of the dataset to be read.

    Returns
    -------
    Raw performance dataframe
    """

    df = pd.read_csv(file_name, sep='|', names=col_per, dtype=perf_type_map, usecols=col_per_subset, index_col=False,
                     nrows=lines_to_read)
    if ref_year != None:
        df = df[df.MonthRep.str.contains('|'.join(ref_year))]
    # Add lagged deliquincy payment value based on CLDS
    df['CLDS'] = df.CLDS.replace('X', '1').astype('float')
    df.loc[df.CLDS == 0.0, 'Arrears'] = 0
    df.loc[df.CLDS != 0.0, 'Arrears'] = 1
    df['Arrears_3m'] = df['Arrears'].rolling(min_periods=3, window=3).apply(
        lambda x: x.sum() if x.sum() < 3 else 0, raw=True).astype('category')
    df['Arrears_6m'] = df['Arrears'].rolling(min_periods=6, window=6).apply(
        lambda x: x.sum() if x.sum() < 6 else 0, raw=True).astype('category')
    df['Arrears_9m'] = df['Arrears'].rolling(min_periods=9, window=9).apply(
        lambda x: x.sum() if x.sum() < 9 else 0, raw=True).astype('category')
    df['Arrears_12m'] = df['Arrears'].rolling(min_periods=12, window=12).apply(
        lambda x: x.sum() if x.sum() < 12 else 0, raw=True).astype('category')

    return df


def create_12mDefault(date, perf_df):
    """
    Create the 12 month forward looking default flag.
    Parameters
    ----------
    date: Snapshot date
    perf_df: Performance dataframe
    Returns
    -------
    Raw observation dataframe
    """
    cur_date = dt.datetime.strptime(date, '%m/%d/%Y').date()
    # Fix the IDs in the observation set by fixing their reporting date AND requiring that the files are healthy.
    obs_df = perf_df[(perf_df.MonthRep == date)
                     &
                     ((perf_df.CLDS == 0.0) |
                      (perf_df.CLDS == 1.0) |
                      (perf_df.CLDS == 2.0)
                      )
                     ]
    obs_ids = obs_df.LoanID
    # Load only the observation IDs in the performance frame initially.
    pf = perf_df[perf_df.LoanID.isin(obs_ids)]

    # Create the 12 month forward looking list of dates
    date_list = []
    for i in np.arange(0, 12):
        if cur_date.month == 12:
            month = 1
            year = cur_date.year + 1
        else:
            month = cur_date.month + 1
            year = cur_date.year
        next_date = dt.datetime(year, month, cur_date.day)
        date_list.append(next_date.strftime('%m/%d/%Y'))
        cur_date = next_date

    # Find the LoanIDs of those loans where a default appears in our 12 month forward looking period.
    pf_obs = perf_df[perf_df.MonthRep.isin(date_list)]
    pf_obs_defaults = pf_obs[
        (pf_obs.CLDS != 0.0) &
        (pf_obs.CLDS != 1.0) &
        (pf_obs.CLDS != 2.0)
        ].LoanID

    pf_obs_defaults = pf_obs_defaults.drop_duplicates(keep='last').values
    df = obs_df
    df['Default'] = 0
    df.loc[df['LoanID'].isin(pf_obs_defaults), 'Default'] = 1

    return df


def remove_default_dupl(observ_df):
    """
    Remove observations with more than one default in observation dataframe
    Parameters
    ----------
    observ_df: observation dataframe

    Returns
    -------
    final observation dataframe
    """
    # Check in observation frame for Loans with more than one defaults per loan ID:
    dft_per_loan = observ_df.groupby('LoanID').Default.sum()
    # List of Loan IDs with more than 1 default:
    dft_per_loan_ids = dft_per_loan[dft_per_loan > 1].index.tolist()
    # Dataframe of loans with more than 1 default:
    dft_per_loan_df = observ_df[observ_df.LoanID.isin(dft_per_loan_ids)]
    # Remove Loans with default flag=0 (cured or not yet in default) and keep 'first' appearing default:
    dft_per_loan_df = dft_per_loan_df[dft_per_loan_df.Default > 0].drop_duplicates(subset='LoanID', keep='first')
    # Healthy dataframe without any loans with more than 1 default
    healthy_frame = observ_df.drop(observ_df[observ_df.LoanID.isin(dft_per_loan_ids)].index)
    # Concat healthy frame and frame containing the filtered loans with
    observs_df_new = pd.concat([healthy_frame, dft_per_loan_df])

    return observs_df_new


def select_sample(observ_df):
    '''
    Select randomly 1/8 of the accounts from each of the 8 quarterly snapshot; an account should appear only once. 
    This way, the final sample will have an even mix of each quarter and will be equivalent size of the portfolio 
    on average over the 2 years.
    Parameters
    ----------
    observ_df: observation dataframe

    Returns
    -------

    '''
    snapshots = observ_df.MonthRep.unique()
    # Store the size of (in case of 8 snapshot dates) 1/8 of the dataset, divided by 8 again to get the size of 1/8 of a snapshot set. 
    # Use this to sample from every snapshot set to come to a final df that contains the 1/8 of the original size and has equal
    # contribution of every quarter/snapshot moment. 
    i = int(observ_df.shape[0] / len(snapshots) / len(snapshots))
    l = []
    for d in snapshots:
        l.append(observ_df[observ_df.MonthRep == d].sample(n=i, replace=False, random_state=1))
    df = pd.concat(l)
    return df

def sample_wo_duplicates(observ_df):
    """
    Sampling without duplicates.
    Parameters
    ----------
    observ_df: the complete observation frame

    Returns
    -------
    Sampled frame
    """
    snapshots = pre_frame.MonthRep.unique()
    snapshots_dates = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in snapshots]
    sample_list = []
    loanids_list = []
    for d in sorted(snapshots_dates)[::-1]:  # Backward looking
        snap_df = pre_frame[pre_frame.MonthRep == d.strftime("%m/%d/%Y")]
        i = int(snap_df.shape[0] / len(snapshots))
        #print('value of 1/8 of the snapshote dataframe= ', i)
        j = len(snap_df)
        # Drop duplicates:
        if loanids_list != None:
            #print('Number of duplicates to kick out= ', snap_df.LoanID.isin(loanids_list).sum())
            snap_df = snap_df[~snap_df.LoanID.isin(loanids_list)]
            #print('Check', j - len(snap_df))
        # Sample
        sampled_df = snap_df.sample(n=i, replace=False, random_state=1)
        # print('Length of sampled df=', len(sampled_df))
        sample_list.append(sampled_df)
        loanids_list.extend(sampled_df.LoanID.unique().tolist())
    agg_sample_df = pd.concat(sample_list)
    return agg_sample_df

def traintest_split(observation_frame, testsize=0.2):
    X = observation_frame.drop('Default', axis=1)
    Y = observation_frame.Default
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, random_state=1)
    return X_train, X_test, y_train, y_test

df = pd.read_csv('Data/Performance_HARP.txt', sep='|', names=col_per, dtype=perf_type_map, usecols=col_per_subset, index_col=False,
                     nrows=None)

"""
    Exploratory analysis:
    Remove number of dates where we do not count any defaults based on CLDS >= 3.
"""
# exclusion_dates = ['04/01/2009', '05/01/2009', '06/01/2009', '07/01/2009', '08/01/2009']
# #df = df[~df['date'].isin(exclusion_dates)]
#
# df['CLDS'] = df.CLDS.replace('X', '1').astype('float')
#
# missing_LoanID = df['LoanID'].isna().sum()
# missing_CLDS = df['CLDS'].isna().sum()
#
# df_count_pop = df[['LoanID','MonthRep']].groupby(["MonthRep"]).count()
# df_count_def = df[df.CLDS >= 3].groupby(["MonthRep"]).count().drop("CLDS", axis=1)
#
# snapshots_without_default = np.setdiff1d(df_count_pop.index.values.tolist(),df_count_def.index.values.tolist())



# if __name__ == "__main__":
#     # Read the file Performance_HARP.txt: http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html
#     performance_frame = read_file(file_name='Data/Performance_HARP.txt', ref_year=['2016', '2017', '2018'],
#                                   lines_to_read=1e6)
#     # Define your snapshot dates for your observation frame:
#     date_list = ['03/01/2016', '06/01/2016', '09/01/2016', '12/01/2016', '03/01/2017', '06/01/2017', '09/01/2017',
#                  '12/01/2017']
#     pre_frame = pd.concat([create_12mDefault(d, performance_frame) for d in date_list])
#     # Remove observations with several defaults:
#     pre_frame = remove_default_dupl(pre_frame)
#     # Sampling
#     observation_frame = sample_wo_duplicates(pre_frame)
#     # observation_frame = select_sample(pre_frame)
#     # Train/test split
#     X_train, X_test, y_train, y_test = traintest_split(observation_frame)
