
# coding: utf-8

# In[1]:


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
import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, f1_score, average_precision_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Import datasets, select features and define the default-flag column.
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


# In[19]:


param = list(eta = 0.01,  objective="binary:logistic",subsample=0.9)
model = XGBClassifier(param)


# In[2]:


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


# In[3]:


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


# In[4]:


def traintest_split(observation_frame, testsize=0.2):
    X = observation_frame.drop('Default', axis=1)
    Y = observation_frame.Default
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=testsize, random_state=1)
    return X_train, X_test, y_train, y_test


# In[5]:


def create_observation_frame():

    # Define your snapshot dates for your observation frame:
    date_list = ['03/01/2016', '06/01/2016', '09/01/2016', '12/01/2016', '03/01/2017', '06/01/2017', '09/01/2017',
                 '12/01/2017']

    pf = pd.read_csv(file_name, sep='|', names=col_per, index_col=False, nrows=lines_to_read)

    pf['CLDS'] = pf.CLDS.replace('X', '1').astype('float')

    observation_frame = pd.concat([create_12mDefault(d, pf) for d in date_list])

    observation_frame = observation_frame[observation_frame.CAUPB.notnull()]
    observation_frame = observation_frame[observation_frame.AdMonthsToMaturity.notnull()]
    return observation_frame


# # Start From this cell

# In[6]:


def get_na_feat(df):
    na_columns = df.columns[df.isnull().any()]
    return na_columns


# In[7]:


def get_cat_feat(df):
    cat_feat = df.select_dtypes(include=['object']).columns
    return cat_feat

def get_num_feat(df):
    num_feat = df.select_dtypes(exclude=['object']).columns
    return num_feat


# In[8]:


def label_encode(df):
    df = df.apply(LabelEncoder().fit_transform)
    return df

def one_hot_encode(df):
    enc = OneHotEncoder(handle_unknown='ignore')
    df = enc.transform(df)
    return df


# In[9]:


def normalize(df):
    df_norm = df
    df_norm = df_norm.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return df_norm


# In[10]:


def preprocess(df):
    # DROP FEATURES WITH NA VALUES
    na_columns = get_na_feat(df)
    df = df.drop(na_columns,axis=1)
    
    df = df.drop('LoanID', axis=1)
    df = df.drop('ModFlag', axis=1)
    
    # FIND THE CATEGORICAL FEATURES
    cat_feat = get_cat_feat(df)

    for cat in cat_feat:
        df[cat] = LabelEncoder().fit_transform(df[cat])
        
    df = normalize(df)
    
    return df


# In[11]:


def make_balanced_df(X,y):

    sm = SMOTE()

    X_cols = X.columns
    X, y = sm.fit_sample(X, y) # fit_sample takes a dataframe, but returns an array. 
    (X, y) = (pd.DataFrame(X, columns=X_cols), pd.Series(y))

    return (X,y)


# In[34]:


def cross_validation():
    # DIVIDE THE DATA IN 10 STRATIFIED FOLDS
    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(X, y)
    
    
    
    for train_index, test_index in skf.split(X, y):
    
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        print(y_train.isnull().values.any())

        # fit model on training data
        model.fit(X_train, y_train)

        # make predictions for test data
        y_pred = model.predict_proba(X_test)
        # temp = pd.concat([X_test,y_pred], axis=1)
        print(y_pred)
        predictions = model.predict(X_test)

        # evaluate predictions
        auc = roc_auc_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0), "|| AUC: %.2f%%" % (auc * 100.0), "|| F1 - Score: %.2f%%" % (f1 * 100.0))


# In[35]:


def test_validation(X,X_val,y,y_val):
    X_val = preprocess(X_val)
    model.fit(X, y)
    y_pred = model.predict_proba(X_val)
    predictions = model.predict(X_val)
    
    auc = roc_auc_score(y_val, predictions)
    print("AUC: " + str(auc))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))


# In[36]:


if __name__ == '__main__':
    file_name='C:/Users/bebxadvberb/Documents/AI/Performance_HARP.txt'
    lines_to_read = 1e5
    
    observation_frame = create_observation_frame()
    X_train, X_val, y_train, y_val = traintest_split(observation_frame)
    X,y = X_train, y_train
    
    X = preprocess(X)
    X,y = make_balanced_df(X,y)
    
    cross_validation()


# In[37]:


if __name__ == '__main__':
    test_validation(X,X_val,y,y_val)

