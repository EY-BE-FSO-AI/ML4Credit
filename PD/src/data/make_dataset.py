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
# BCS                  = Borrower Credit Score (A)
# channel              = Channel (A)
# CBCS                 = Co-Borrower Credit Score (A)
# seller_name          = Seller Name (A)
# OIR                  = Original Interest Rate (A)
# OUPB                 = Original Unpaid Principal Balance (A)
# OLT                  = Original Loan Term (A)
# ODate                = Origination Date (A)
# FPDate               = First Payment Date (A)
# OLTV                 = Original Loan-to-Value (A)
# OCLTV                = Original Combined Loan-to-Value (A)
# nbr_borr             = Number of borrowers (A)
# DTI                  = Debt-To-Income ratio (A)
# FTHBI                = First-Time Home Buyer Indicator (A)
# LoanPurp             = Loan Purpose (A)
# PropType             = Property Type (A)
# nb_units             = number of units (A)
# OccupStatus          = Occupancy status (A)
# PropState            = Property State (A)
# ZIP                  = ZIP (A)
# MIP                  = Mortgage Insurance Percentage (A)
# ProdType             = Product Type (A)
# MIT                  = Mortgage Insurance Type (A)
# RMI                  = Relocation Mortgage Indicator (A)

"""
    Import statements 
"""

import pandas as pd
import numpy as np
import datetime as dt

# Import datasets, select features and define the default-flag collumn.
col_per = ['LoanID', 'MonthRep', 'Servicer', 'CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity',
           'AdMonthsToMaturity', 'MaturityDate', 'MSA', 'CLDS', 'ModFlag', 'ZeroBalCode', 'ZeroBalDate',
           'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'ForeclosureCosts', 'PPRC', 'AssetRecCost', 'MHEC',
           'ATFHP', 'NetSaleProceeds', 'CreditEnhProceeds', 'RPMWP', 'OFP', 'NIBUPB', 'PFUPB', 'RMWPF',
           'FPWA', 'ServicingIndicator']

col_acq = ['LoanID','channel', 'seller_name','OIR','OUPB','OLT','ODate','FPDate','OLTV','OCLTV','nbr_borr','DTI','BCS',
           'FTHBI', 'LoanPurp', 'PropType','nb_units','OccupStatus','PropState','ZIP','MIP','ProdType', 'CBCS','MIT','RMI']

# Python will guess the datatypes not specified in the map function, for dates the dtype will be 'object'. (hence: here all dates)
# If an expected integer variables contains NaN values it will be set to 'float32'
perf_type_map = {'LoanID': 'int64', 'Servicer': 'category', 'CurrInterestRate': 'float32', 'CAUPB': 'float32',
                 'LoanAge': 'int64', 'MonthsToMaturity': 'int64', 'AdMonthsToMaturity': 'float32', 'MSA': 'category',
                 'CLDS': 'category', 'ModFlag': 'category', 'ZeroBalCode': 'float32', 'ForeclosureCosts': 'float32',
                 'PPRC': 'float32', 'AssetRecCost': 'float32', 'MHEC': 'float32', 'ATFHP': 'float32',
                 'NetSaleProceeds': 'float32', 'CreditEnhProceeds': 'float32', 'RPMWP': 'float32', 'OFP': 'float32',
                 'NIBUPB': 'float32', 'PFUPB': 'float32', 'RMWPF': 'category', 'FPWA': 'float32',
                 'ServicingIndicator': 'category'}

extended_selec_per = ['LoanID', 'MonthRep', 'Servicer', 'CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity',
           'AdMonthsToMaturity', 'MaturityDate', 'MSA', 'CLDS', 'ModFlag', 'ZeroBalCode', 'ZeroBalDate',
           'LastInstallDate','FPWA', 'ServicingIndicator']

col_per_subset = extended_selec_per

def read_file(file_name, ref_year, col_names, use_cols, type_map):
    df = pd.read_csv(file_name, sep='|', names=col_names, dtype=type_map, usecols=use_cols,
                     index_col=False)
    if ref_year != None:
        df = df[df.MonthRep.str.contains('|'.join(ref_year))]
    return df

def create_arrears(df):
    """
    Create in arrears variables
    Parameters
    ----------
    df:

    Returns
    -------
    Raw performance dataframe
    """
    df.loc[df.CLDS == '0', 'Arrears'] = 0
    df.loc[df.CLDS != '0', 'Arrears'] = 1
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
    # Treat missing as default (conservative)
    perf_df['CLDS'] = perf_df.CLDS.replace('X', '1')
    # Fix the IDs in the observation set by fixing their reporting date AND requiring that the files are healthy.
    obs_df = perf_df[(perf_df.MonthRep == date)
                     &
                     ((perf_df.CLDS == "0") |
                      (perf_df.CLDS == "1") |
                      (perf_df.CLDS == "2")
                      )
                     ]

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
        (pf_obs.CLDS != "0") &
        (pf_obs.CLDS != "1") &
        (pf_obs.CLDS != "2")
        ].LoanID

    pf_obs_defaults = pf_obs_defaults.drop_duplicates(keep='last').values
    obs_df['Default'] = 0
    obs_df.loc[obs_df['LoanID'].isin(pf_obs_defaults), 'Default'] = 1

    return obs_df


def remove_default_dupl(observ_df):
    """
    Remove observations with more than one default in observation dataframe
    (In other words remove cured observations)
    Parameters
    ----------
    observ_df: observation dataframe

    Returns
    -------
    final observation dataframe
    """
    dft_per_loan = observ_df.groupby('LoanID')
    res = []
    for group, grouped in dft_per_loan:
        try:
            #Get first appearance of a default
            change_index = grouped[grouped.Default == 1].drop_duplicates(subset='LoanID', keep='first').index.values[0]
            #Drop the rest
            res.append( grouped.loc[:change_index, :] )
        except:
            res.append(grouped)
    observs_df_new = pd.concat(res)
    return observs_df_new

def run_defaultflag(file_name, ref_year=['2017'], use_cols=['LoanID','CLDS']):
    '''
    As it says.
    Returns
    -------
    '''
    dflt_frame = read_file(file_name, ref_year, use_cols)
    res = []
    for d in dflt_frame.MonthRep.unique():
        res.append(create_12mDefault(d, dflt_frame))
        print('Date {d} done.'.format(d=d))
    dflt_frame = pd.concat(res)
    print('Removing duplicate defaults...')
    dflt_frame = remove_default_dupl(dflt_frame)
    return dflt_frame

if __name__ == "__main__":
    # Performance_HARP.txt: http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html
    use_years = None # e.g. ['2009', '2010', '2011']
    # Create the default flag for the whole dataset
    dflt_frame = run_defaultflag(file_name='Data/Performance_HARP.txt', ref_year=use_years, col_names=col_per,
                                 use_cols=['LoanID','CLDS','MonthRep'], type_map=perf_type_map) #Or "A" dataframe
    # Read the performance frame:
    performance_frame = read_file(file_name='Data/Performance_HARP.txt', ref_year=use_years, col_names=col_per,
                                  use_cols=col_per_subset, type_map=perf_type_map) #Or "B" dataframe
    # Join both dataframe
    performance_frame = pd.merge(performance_frame, dflt_frame[['LoanID','MonthRep', 'Default']], on=['LoanID','MonthRep'])
    # Read the acquisition frame:
    acquisition_frame = read_file(file_name='Data/Acquisition_HARP.txt', ref_year=None, col_names=col_acq,
                                  use_cols=col_acq, type_map=None)
    # Full dataset:
    full_frame = pd.merge(performance_frame,acquisition_frame, on='LoanID')
