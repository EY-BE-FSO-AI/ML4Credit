"""
Glossary mapping
"""

#LoanID               = Loan Identifier (A,P)
#MonthRep             = Monthly Reporting Period (P)
#Servicer             = Servicer Name (P)
#CurrInterestRate     = CURRENT INTEREST RATE (P)
#CAUPB                = CURRENT ACTUAL UNPAID PRINCIPAL BALANCE (P)
#LoanAge              = Loan Age (P)
#MonthsToMaturity     = Remaining Months to Legal Maturity (P)
#AdMonthsToMaturity   = ADJUSTED REMAINING MONTHS TO MATURITY (P)
#MaturityDate         = Maturity Date (P)
#MSA                  = Metropolitan Statistical Area (P)
#CLDS                 = Current Loan Delinquency Status (P)
#ModFlag              = Modification Flag (P)
#ZeroBalCode          = Zero Balance Code (P)
#ZeroBalDate          = Zero Balance Effective Date(P)
#LastInstallDate      = LAST PAID INSTALLMENT DATE
#ForeclosureDate      = FORECLOSURE DATE
#DispositionDate      = DISPOSITION DATE
#ForeclosureCosts     = FORECLOSURE COSTS (P)
#PPRC                 = Property Preservation and Repair Costs (P)
#AssetRecCost         = ASSET RECOVERY COSTS (P)
#MHEC                 = Miscellaneous Holding Expenses and Credits (P)
#ATFHP                = Associated Taxes for Holding Property (P)
#NetSaleProceeds      = Net Sale Proceeds (P)
#CreditEnhProceeds    = Credit Enhancement Proceeds (P)
#RPMWP                = Repurchase Make Whole Proceeds(P)
#OFP                  = Other Foreclosure Proceeds (P)
#NIBUPB               = Non-Interest Bearing UPB (P)
#PFUPB                = PRINCIPAL FORGIVENESS UPB (P)
#RMWPF                = Repurchase Make Whole Proceeds Flag (P)
#FPWA                 = Foreclosure Principal Write-off Amount (P)
#ServicingIndicator   = SERVICING ACTIVITY INDICATOR (P)

"""
    Import statements 
"""

import pandas as pd
import numpy as np
import re
import traceback
import dask.dataframe as dd
from dask.distributed import Client
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats.stats as stats
import statsmodels.api as  sm
import pandas.core.algorithms as algos
import datetime as dt
from sklearn.model_selection import train_test_split


#Import datasets, select features and define the default-flag collumn.
col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
              'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
              'LastInstallDate','ForeclosureDate','DispositionDate', 'ForeclosureCosts', 'PPRC','AssetRecCost','MHEC',
              'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
              'FPWA','ServicingIndicator'] 

#Python will guess the datatypes not specified in the map function, for dates the dtype will be 'object'. (hence: here all dates)
#If an expected integer variables contains NaN values it will be set to 'float32'
perf_type_map = {'LoanID' : 'int64', 'Servicer' : 'category', 'CurrInterestRate' : 'float32', 'CAUPB' : 'float32',
                 'LoanAge' : 'int64', 'MonthsToMaturity' : 'int64', 'AdMonthsToMaturity' : 'float32', 'MSA' : 'category',
                 'CLDS' : 'category', 'ModFlag' : 'category', 'ZeroBalCode' : 'float32','ForeclosureCosts' : 'float32', 
                 'PPRC' : 'float32', 'AssetRecCost' : 'float32', 'MHEC' : 'float32', 'ATFHP' : 'float32', 
                 'NetSaleProceeds' : 'float32', 'CreditEnhProceeds' : 'float32', 'RPMWP' : 'float32', 'OFP' : 'float32',
                 'NIBUPB' : 'float32','PFUPB' : 'float32', 'RMWPF' : 'category', 'FPWA' : 'float32', 'ServicingIndicator' : 'category'}


extended_selec_per = col_per

col_per_subset =  extended_selec_per 
    
def read_file(file_name, ref_year, lines_to_read=None):
    """
    Read file in function to avoid memory issues
    + Add lagged payment variables
    Parameters
    ----------
    file_name: Path name of the file;
    ref_year: Specify the list of years to be read;
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
    for i in np.arange(0,12):
        if cur_date.month == 12:
            month = 1
            year = cur_date.year + 1
        else:
            month = cur_date.month + 1
            year = cur_date.year
        next_date = dt.datetime(year , month, cur_date.day)
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
    observ_df

    Returns
    -------

    '''
    snapshots = observ_df.MonthRep.unique()
    i = int(observ_df.shape[0] / len(snapshots) / 8)
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
    Y = observation_frame.Default.to_frame()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, random_state=1)
    return X_train, X_test, y_train, y_test

# Read the file Performance_HARP.txt: http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html
performance_frame = read_file(file_name='Data/Performance_HARP.txt', ref_year=['2016','2017','2018'])
# Define your snapshot dates for your observation frame:
date_list = ['03/01/2016', '06/01/2016', '09/01/2016', '12/01/2016', '03/01/2017', '06/01/2017', '09/01/2017', '12/01/2017']
pre_frame = pd.concat([create_12mDefault(d, performance_frame) for d in date_list])
# Remove observations with several defaults:
pre_frame = remove_default_dupl(pre_frame)
# Sampling
post_frame = sample_wo_duplicates(pre_frame)
# Train/test split
X_train, X_test, y_train, y_test = traintest_split(post_frame)
#observation_frame = X_train


'''
 Variable selection and binning
'''


# Treatment of missing values:
#Change MatruityDate to maturity year
X_train['maturity_year'] = X_train.MaturityDate.apply(lambda x: x[-4:]).astype('int')
X_train = X_train.drop(labels=['MaturityDate'], axis=1)

# Add coarsed CLDS (<3)
X_train.CLDS = X_train.CLDS.astype('category')

# Remove variables with more than 20% missing values
missing_tolerance = 20 #percent
col_del = X_train.columns[X_train.isnull().sum() * 100 / len(X_train) > missing_tolerance]
X_train = X_train.drop(labels=col_del, axis=1)

# List of numeric and categorical variables
X_cont = X_train.select_dtypes(include=['int64','float32','float64']).columns.values
X_cont = np.delete(X_cont, np.where(X_cont == 'Arrears'))
X_cat = X_train.select_dtypes(include=['category']).columns.values
X_cat = np.delete(X_cat, np.where(X_cat == 'MSA')) # Remove MSA for the moment

# Replace missing values: numeric/continious -> mean, category -> mode
col_fill_mean = X_train[X_cont].columns[X_train[X_cont].isnull().mean() > 0]
X_train[col_fill_mean] = X_train[col_fill_mean].fillna(X_train[col_fill_mean].mean())
col_fill_mode = X_train[X_cat].columns[X_train[X_cat].isnull().sum() * 100 / len(X_train) > 0]
X_train[col_fill_mode] = X_train[col_fill_mode].fillna(X_train[col_fill_mode].mode())

#Additional:
X_train.Arrears_6m = X_train.Arrears_6m.astype('int').replace(5, 4).astype('category')
X_train.Arrears_6m = X_train.Arrears_6m.astype('int').replace(4, 3).astype('category')
X_train.Arrears_6m = X_train.Arrears_6m.astype('int').replace(3, 2).astype('category')

# --> develop  treatment of extreme values



""" 
    WOE 
    
    https://github.com/Sundar0989/WOE-and-IV/blob/master/WOE_IV.ipynb
    
"""


def WOE(Y, X, var_name="VAR", binning=True, nb_quantiles=10):
    """
    Bin variable and compute WoE/IV.
    Parameters
    ----------
    Y: response variable (1d pandas)
    X: predictor variable (1d pandas)
    var_name: predictor (X) variable name
    binning: True/False
    nb_quantiles: Number of equal-sized bins (pandas qcut)

    Returns
    -------

    """

    if binning:
        X = pd.qcut(x=X, q=np.arange(0, 1.1, 1/nb_quantiles), duplicates='drop')
        # X = pd.qcut(x=X.rank(method='first'), q=np.arange(0, 1.1, 1/nb_quantiles), duplicates='drop')

    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = var_name
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    d3["BIN"] = d3.MIN_VALUE #d3.MIN_VALUE.map(str)
    d3 = d3.drop(labels=['MIN_VALUE', 'MAX_VALUE'], axis=1)
    cols = d3.columns.to_list()
    d3 = d3[cols[:1] + cols[-1:] + cols[1:-1]]

    return (d3)


# define a binning function
def mono_bin(Y, X, n=20, m=3, var_name="VAR"):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    r = 0
    while np.abs(r) < 1:
        try:
            d1 = pd.DataFrame({"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.qcut(notmiss.X, n)}) #rank variables
            d2 = d1.groupby('Bucket', as_index=True)
            r, p = stats.spearmanr(d2.mean().X, d2.mean().Y) #Use Spearman corr for ranked var(not pearson)
            n = n - 1
        except Exception as e:
            n = n - 1

    if len(d2) == 1:
        n = m
        bins = algos.quantile(notmiss.X, np.linspace(0, 1, n))
        if len(np.unique(bins)) == 2:
            bins = np.insert(bins, 0, 1)
            bins[1] = bins[1] - (bins[1] / 2)
        d1 = pd.DataFrame(
            {"X": notmiss.X, "Y": notmiss.Y, "Bucket": pd.cut(notmiss.X, np.unique(bins), include_lowest=True)})
        d2 = d1.groupby('Bucket', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["MIN_VALUE"] = d2.min().X
    d3["MAX_VALUE"] = d2.max().X
    d3["COUNT"] = d2.count().Y
    d3["EVENT"] = d2.sum().Y
    d3["NONEVENT"] = d2.count().Y - d2.sum().Y
    d3 = d3.reset_index(drop=True)

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = var_name
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3["BIN"] = "(" + d3.MIN_VALUE.map(str) + ", " + d3.MAX_VALUE.map(str) + "]"
    d3["BIN_low"] = d3.MIN_VALUE
    d3["BIN_up"] = d3.MAX_VALUE
    d3 = d3.drop(labels=['MIN_VALUE', 'MAX_VALUE'], axis=1)
    cols = d3.columns.to_list()
    d3 = d3[cols[:1] + cols[-1:] + cols[1:-1]]

    return (d3)


def char_bin(Y, X):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE': np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True)

    d3["EVENT_RATE"] = d3.EVENT / d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT / d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT / d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT / d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)

    return (d3)


def data_vars(df1, target, max_bin = 20, force_bin = 3):
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]

    x = df1.dtypes.index
    count = -1

    for i in x:
        if i.upper() not in (final.upper()):
            if np.issubdtype(df1[i], np.number) and len(Series.unique(df1[i])) > 2:
                conv = mono_bin(target, df1[i], n=max_bin, m=force_bin)
                conv["VAR_NAME"] = i
                count = count + 1
            else:
                conv = char_bin(target, df1[i])
                conv["VAR_NAME"] = i
                count = count + 1

            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv, ignore_index=True)

    iv = pd.DataFrame({'IV': iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return (iv_df, iv)


def woeiv_results(df, dfltvar, cont_var, cat_var):
    """
    Aggregate IV/WoE results in dataframes
    Parameters
    ----------
    df: observation dataframe
    cont_var: continious variables
    cat_var: categorical variables
    dfltvar: column of default

    Returns IV df, fine_WoE df and coarse_WoE df
    -------

    """
    # Fine Classing
    fine_cont_woe = pd.concat([WOE(dfltvar, df[x], var_name=x) for x in cont_var])
    fine_cont_iv = fine_cont_woe[['VAR_NAME', 'IV']].drop_duplicates()
    fine_cat_woe = pd.concat([WOE(dfltvar, df[x], var_name=x, binning=False) for x in cat_var])
    fine_cat_iv = fine_cat_woe[['VAR_NAME', 'IV']].drop_duplicates()
    # Coarse Classing (only for continious variables
    coarse_cont_woe, coarse_cont_iv = data_vars(df[cont_var], dfltvar)

    final_iv = pd.concat([pd.merge(fine_cont_iv, coarse_cont_iv, how='inner', on='VAR_NAME'),
                          pd.merge(fine_cat_iv, fine_cat_iv, how='inner', on='VAR_NAME')])
    final_iv.columns = ['VAR_NAME','fine_IV', 'coarse_IV']
    final_fine_woe = pd.concat([fine_cont_woe, fine_cat_woe])
    final_coarse_woe = pd.concat([coarse_cont_woe, fine_cat_woe])

    return final_iv, final_fine_woe, final_coarse_woe


iv, fine_woe, coarse_woe = woeiv_results(df=X_train, dfltvar=y_train['Default'], cont_var=X_cont, cat_var=X_cat) # Drop non-coarsed CLDS in coarse df

#selection rule IV > 0.10
inputs = iv[iv.coarse_IV > 0.1].VAR_NAME.tolist()
covariates = X_train[inputs]


'''
Check for Collinearity and Multicollineairty 
'''

# Collineairty - Correlation:
covariates.ModFlag = covariates.ModFlag.replace(['Y','N'], [1,0]) # convert to numeric
covariates[covariates.select_dtypes('category').columns] = covariates.select_dtypes('category').astype('float') #convert to numeric
covariates_corr = covariates.corr()

# Multicollineairty - VIF
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

def variance_inflation_factors(exog_df, addconst=False):
    '''
    Parameters
    ----------
    exog_df : dataframe, (nobs, k_vars)
        design matrix with all explanatory variables, as for example used in
        regression.
    addconst: Add constant to dataframe

    Returns
    -------
    vif : Series
        variance inflation factors
    '''
    if addconst:
        exog_df = add_constant(exog_df)
    vif = {}
    for col in exog_df:
        reg_rsq = OLS(exog_df[col].values, exog_df.drop(col, axis=1).values).fit().rsquared
        vif[col] = 1 / (1 - reg_rsq)
    return vif

#Compute VIF:
vif = variance_inflation_factors(covariates, True)
vif = pd.DataFrame(vif.values(), index=vif.keys(), columns=['VIF'])

#Drop variables with VIF > 5 and compute VIF second time
keep_var = vif[vif < 5].dropna().index.tolist()
vif_2nd = variance_inflation_factors(covariates[keep_var], addconst=True) # No VIF > 5, then good to go
vif_2nd = pd.DataFrame(vif_2nd.values(), index=vif_2nd.keys(), columns=['VIF_2nd'])

#Select covariates with VIF < 5
covariates = covariates[vif_2nd.drop('const', axis=0).index.tolist()]


'''
Logistic regression
'''

#Logistic regression
#equivalent, different library: sk.linear_model.LinearRegression()
def run_logit(y, X):
    model = sm.Logit(y, X)
    rslt = model.fit()
    print(rslt.summary2())
    return model, rslt

# Use WOE values of continious binned variables:
def map_var_to_woe(covariates, coarse_woe):
    covar_bins = coarse_woe[coarse_woe.VAR_NAME.isin(covariates.columns)].groupby('VAR_NAME').apply(
        lambda df: pd.IntervalIndex.from_arrays(df.BIN_low.astype('int'), df.BIN_up.astype('int'), closed='both')
        if df.name in X_cont
        else pd.IntervalIndex.from_breaks(df.index)).dropna()

    binned_covar = {x: pd.cut(covariates[x], covar_bins[x]).cat.codes if x in X_cont
    else pd.cut(covariates[x], covar_bins[x]).cat.codes + 1 for x in covariates.columns.tolist()}
    woe_covar = {}
    var_names = []
    for j in binned_covar.keys():
        woe_values = coarse_woe.groupby('VAR_NAME').WOE.apply(lambda df: df.values)[j]
        woe_covar[j] = binned_covar[j].apply(lambda x : woe_values[x])
        var_names.append( j )

    new_covariates = covariates.drop(var_names, axis=1).join( pd.DataFrame(woe_covar), how='inner' )
    return new_covariates

covariates_woe = map_var_to_woe(covariates, coarse_woe)
# We drop 3m arrears as it has too much correlation/collinearity with other variables.
covariates_woe.drop('Arrears_3m', axis=1, inplace=True)

# Logistic regression
logit_model, results = run_logit(y_train, covariates_woe)

#Drop variable with p-value > 0.01:
cov_proc = results.pvalues[results.pvalues < 0.01].index.tolist()
logit_model_proc, results_proc = run_logit(y_train, covariates_woe[cov_proc])

def run_model_tests(logit_res, x_test, y_test, covar_list):
    # Create
    X_test_proc = x_test[covar_list]
    X_test_proc.ModFlag = X_test_proc.ModFlag.replace(['Y','N'], [1,0]) # convert to numeric
    X_test_proc[X_test_proc.select_dtypes('category').columns] = X_test_proc.select_dtypes('category').astype('float') #convert to numeric
    X_test_proc = map_var_to_woe(X_test_proc, coarse_woe)
    #Predict
    predicted = logit_res.predict(X_test_proc)

    #Get the crude binning, auc, gini:
    crude_bins = pd.qcut(x=predicted, q=10, duplicates='drop').map(str).to_frame(name='crude_bins')
    #crude_bins = pd.cut(predicted, np.arange(0.0, 1.1, 1/10)).map(str).to_frame(name='crude_bins')
    df_joined = x_test.join([crude_bins, y_test], how='inner')
    sum_actual_dflt = df_joined.groupby('crude_bins').Default.sum()
    sum_total_accts = df_joined.groupby('crude_bins')['LoanID'].count()
    sum_dflt_rate = (sum_actual_dflt / sum_total_accts).replace(np.nan, 0)
    good_df = sum_total_accts - sum_actual_dflt
    perc_good = good_df / good_df.sum()
    perc_bad = sum_actual_dflt / sum_actual_dflt.sum()
    cum_good = perc_good.cumsum()
    cum_bad = perc_bad.cumsum()
    distance = abs(cum_good - cum_bad)
    auc_df = ( cum_bad.values[1:] + cum_bad.values[:-1] ) / 2 * ( cum_good.values[1:] - cum_good.values[:-1] )
    analysis_df = pd.concat([sum_actual_dflt, sum_total_accts, sum_dflt_rate, good_df, perc_good, perc_bad, cum_good, cum_bad, distance], axis=1)
    analysis_df_cols = ['Sum_of_actual_dflt', 'Sum_of_Total_Accounts', 'Sum_of_Default_Rate', 'Good', '%Good', '%Bad', 'Cum Good', 'Cum Bad', 'Distance']
    analysis_df.columns = analysis_df_cols
    first_row = pd.DataFrame(np.zeros((1, len(analysis_df_cols))), index=['0'], columns=analysis_df_cols)
    analysis_df = pd.concat([first_row, analysis_df], axis=0)
    auc = np.sum( auc_df )
    gini = 1 - auc
    return analysis_df, auc, gini

analysis_IS, _, gini_IS = run_model_tests(results_proc, X_train, y_train, cov_proc) #Out of sample
analysis_OoS, _, gini_OoS = run_model_tests(results_proc, X_test, y_test, cov_proc) #Out of sample
auc_IS = (1 + gini_IS) / 2
auc_OsS = (1 + gini_OoS) / 2

# Write results to excel
writer = pd.ExcelWriter('classicalPD_IVs.xlsx', engine='xlsxwriter')
iv.to_excel(writer, sheet_name='IV')
coarse_woe.to_excel(writer, sheet_name='Coarse')
fine_woe.to_excel(writer, sheet_name='Fine')
covariates_corr.to_excel(writer, sheet_name='Correlation')
vif.to_excel(writer, sheet_name='VIF')
vif_2nd.to_excel(writer, sheet_name='VIF_2nd')
analysis_OoS.to_excel(writer, sheet_name='AUC_OutofSample')
analysis_IS.to_excel(writer, sheet_name='AUC_InSample')
writer.save()

#Lorenz curve
plt.plot(analysis_OoS['Cum Bad'],analysis_OoS['Cum Good'], label='Out-of-Sample')
plt.plot(analysis_IS['Cum Bad'],analysis_IS['Cum Good'], label='In-Sample')
diag_line = np.linspace(0, 1, len(analysis_IS))
plt.plot(diag_line, diag_line, linestyle='--', c='red')
#plt.text(0.85, 0.05, 'AUC_IS = {s}%'.format(s=np.round(auc_IS*100, 2)), horizontalalignment='center', verticalalignment='center')
#plt.text(0.90, 0.10, 'AUC_OoS = {s}%'.format(s=np.round(auc_OsS*100, 2)), horizontalalignment='center', verticalalignment='center')
plt.xlabel('Cum bad')
plt.ylabel('Cum good')
plt.title('Lorenz curve')
plt.legend()
plt.show()

# Plot K-S graph:
analysis_OoS['Cum Bad'].plot(rot=45)
analysis_OoS['Cum Good'].plot(rot=45)
plt.title('Kolmogorov-Smirnov')
plt.legend()
plt.show()
analysis_IS['Cum Bad'].plot(rot=45)
analysis_IS['Cum Good'].plot(rot=45)
plt.title('Kolmogorov-Smirnov')
plt.legend()
plt.show()

# Create, output AUC
# def compute_auc(xtest, ytest, covar_list, log_results):
#     xtest = xtest[covar_list]
#     xtest.ModFlag = xtest.ModFlag.replace(['Y','N'], [1,0]) # convert to numeric
#     xtest[xtest.select_dtypes('category').columns] = xtest.select_dtypes('category').astype('float') #convert to numeric
#     X_test_proc = map_var_to_woe(xtest, coarse_woe)
#     predicted = log_results.predict(X_test_proc)
#     from sklearn.metrics import roc_auc_score, roc_curve
#     auc = roc_auc_score(y_true=ytest, y_score=predicted)
#     fpr, tpr, thresholds = roc_curve(y_true=ytest, y_score=predicted)
#     print("AUC :", auc)
#     return auc, fpr, tpr
#
# ### Plot ROC, AUC etc.
# auc, fpr, tpr = compute_auc(xtest=X_test, ytest=y_test, covar_list=cov_proc, log_results=results)
# auc, fpr, tpr = compute_auc(xtest=X_train, ytest=y_train, covar_list=cov_proc, log_results=results)
# diag_line = np.linspace(0, 1, len(fpr))
# plt.plot(diag_line, diag_line, linestyle='--', c='red')
# plt.plot(fpr, tpr)
# #plt.plot(thresholds)
# plt.ylabel('True Pos. Rate')
# plt.xlabel('False Pos. Rate')
# plt.text(0.90, 0.05, 'AUC = {s}%'.format(s=np.round(auc*100, 2)), horizontalalignment='center', verticalalignment='center')
# plt.show()


"""
Appendix: Variable exploration, description of the dataset
"""

