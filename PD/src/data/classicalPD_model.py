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
from pandas import Series
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats.stats as stats
import statsmodels.api as  sm
import pandas.core.algorithms as algos
import datetime as dt


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


# Read the file Performance_HARP.txt: http://www.fanniemae.com/portal/funding-the-market/data/loan-performance-data.html
performance_frame = read_file(file_name='Data/Performance_HARP.txt', ref_year=['2016','2017','2018'])
# Define your snapshot dates for your observation frame:
date_list = ['12/01/2016', '03/01/2017', '06/01/2017', '09/01/2017', '12/01/2017']
observation_frame = pd.concat([create_12mDefault(d, performance_frame) for d in date_list])
# Remove observations with several defaults:
observation_frame = remove_default_dupl(observation_frame)



'''
 Variable selection and binning
'''

# Treatment of missing values:
#Group MatruityDate by year
observation_frame['maturity_year'] = observation_frame.MaturityDate.apply(lambda x: x[-4:]).astype('int')
observation_frame = observation_frame.drop(labels=['MaturityDate'], axis=1)

# Add coarsed CLDS (<3)
#observation_frame['CLDS_coarse'] = observation_frame.CLDS.apply(lambda x: x if x == 'X' else 'NAN' if x == 'NAN' else '>3' if int(x) >= 3 else x).astype('category')
observation_frame.CLDS = observation_frame.CLDS.astype('category')

missing_tolerance = 20 #percent
col_del = observation_frame.columns[observation_frame.isnull().sum() * 100 / len(observation_frame) > missing_tolerance]
observation_frame = observation_frame.drop(labels=col_del, axis=1)
X_cont = observation_frame.select_dtypes(include=['int64','float32','float64']).columns
X_cat = observation_frame.select_dtypes(include=['category']).columns

# Replace missing values: continious -> mean, category -> mode
col_fill_mean = observation_frame[X_cont].columns[observation_frame[X_cont].isnull().mean() > 0]
observation_frame[col_fill_mean] = observation_frame[col_fill_mean].fillna(observation_frame[col_fill_mean].mean())  #does mean need to be an integer?
col_fill_mode = observation_frame[X_cat].columns[observation_frame[X_cat].isnull().sum() * 100 / len(observation_frame) > 0]
observation_frame[col_fill_mode] = observation_frame[col_fill_mode].fillna(observation_frame[col_fill_mode].mode())

#Select all categorical and continuous variables.
X_cont = ['CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity', 'AdMonthsToMaturity']# Manual selections
X_cont = np.delete(X_cont, np.where(X_cont == 'Default'))
X_cat = observation_frame.select_dtypes(include=['category']).columns.values
X_cat = np.delete(X_cat, np.where(X_cat == 'MSA')) # Remove MSA for the moment
Y = ['Default']

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
    d3["BIN"] = d3.MIN_VALUE.map(str)
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


def woeiv_results(df, cont_var, cat_var, dfltvar_name='Default'):
    """
    Aggregate IV/WoE results in dataframes
    Parameters
    ----------
    df: observation dataframe
    cont_var: continious variables
    cat_var: categorical variables
    dfltvar_name: Name of default column in df dataframe.

    Returns IV df, fine_WoE df and coarse_WoE df
    -------

    """
    # Fine Classing
    fine_cont_woe = pd.concat([WOE(df[dfltvar_name], df[x], var_name=x) for x in cont_var])
    fine_cont_iv = fine_cont_woe[['VAR_NAME', 'IV']].drop_duplicates()
    fine_cat_woe = pd.concat([WOE(df[dfltvar_name], df[x], var_name=x, binning=False) for x in cat_var])
    fine_cat_iv = fine_cat_woe[['VAR_NAME', 'IV']].drop_duplicates()
    # Coarse Classing (only for continious variables
    coarse_cont_woe, coarse_cont_iv = data_vars(df[cont_var], df[dfltvar_name])

    final_iv = pd.concat([pd.merge(fine_cont_iv, coarse_cont_iv, how='inner', on='VAR_NAME'),
                          pd.merge(fine_cat_iv, fine_cat_iv, how='inner', on='VAR_NAME')])
    final_iv.columns = ['VAR_NAME','fine_IV', 'coarse_IV']
    final_fine_woe = pd.concat([fine_cont_woe, fine_cat_woe])
    final_coarse_woe = pd.concat([coarse_cont_woe, fine_cat_woe])

    return final_iv, final_fine_woe, final_coarse_woe


iv, fine_woe, coarse_woe = woeiv_results(df=observation_frame, cont_var=X_cont, cat_var=X_cat) # Drop non-coarsed CLDS in coarse df

# Write results to excel
writer = pd.ExcelWriter('classicalPD_IVs.xlsx', engine='xlsxwriter')
iv.to_excel(writer, sheet_name='IV')
coarse_woe.to_excel(writer, sheet_name='Coarse')
fine_woe.to_excel(writer, sheet_name='Fine')
writer.save()

'''
Logistic regression
'''
#equivalent, different library: sk.linear_model.LinearRegression()

#selection rule IV > 0.10
inputs = iv[iv.coarse_IV > 0.1].VAR_NAME.tolist()
df_input = observation_frame[inputs]
df_input.ModFlag = df_input.ModFlag.replace(['Y','N'], [True,False])
df_input[df_input.select_dtypes(['category','bool']).columns] = df_input.select_dtypes(['category','bool']).astype('int')
df_input = df_input.drop(labels='CLDS', axis=1)

#logistic regression
logit_model = sm.Logit(observation_frame.Default, df_input.astype(float))
result = logit_model.fit()
print(result.summary2())


"""
Appendix: Variable exploration, description of the dataset
"""
#SellerName
len(observation_frame.SellerName.unique())
observation_frame[observation_frame['SellerName']=='Other'].count
