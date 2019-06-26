import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from functools import reduce
import seaborn as sns

sns.set(font_scale = 2)

### Code by Nicolas/Bram
col_acq = ['LoanID', 'Channel', 'SellerName', 'OrInterestRate', 'OrUnpaidPrinc', 'OrLoanTerm',
           'OrDate', 'FirstPayment', 'OrLTV', 'OrCLTV', 'NumBorrow', 'DTIRat', 'CreditScore',
           'FTHomeBuyer', 'LoanPurpose', 'PropertyType', 'NumUnits', 'OccStatus', 'PropertyState',
           'Zip', 'MortInsPerc', 'ProductType', 'CoCreditScore', 'MortInsType', 'RelMortInd']

extended_selec_acq = ['LoanID', 'OrLTV', 'LoanPurpose', 'DTIRat', 'PropertyType', 'FTHomeBuyer', 'Channel',
                      'SellerName', 'OrInterestRate', 'CreditScore', 'NumBorrow', 'OrDate']

col_acq_subset = extended_selec_acq

col_per = ['LoanID', 'MonthRep', 'Servicer', 'CurrInterestRate', 'CAUPB', 'LoanAge', 'MonthsToMaturity',
           'AdMonthsToMaturity', 'MaturityDate', 'MSA', 'CLDS', 'ModFlag', 'ZeroBalCode', 'ZeroBalDate',
           'LastInstallDate', 'ForeclosureDate', 'DispositionDate', 'ForeclosureCosts', 'PPRC', 'AssetRecCost', 'MHRC',
           'ATFHP', 'NetSaleProceeds', 'CreditEnhProceeds', 'RPMWP', 'OFP', 'NIBUPB', 'PFUPB', 'RMWPF',
           'FPWA', 'ServicingIndicator']

extended_selec_per = ['LoanID', 'MonthsToMaturity', 'CurrInterestRate', 'ForeclosureDate', 'LoanAge', 'CLDS',
                      'MaturityDate', 'MonthRep', 'NetSaleProceeds', 'OFP', 'FPWA', 'CreditEnhProceeds',
                      'RPMWP', 'DispositionDate', 'CurrInterestRate', 'CAUPB', 'ForeclosureCosts', 'MHRC',
                      'AssetRecCost', 'PPRC', 'ATFHP']

col_per_subset = col_per

lines_to_read = None
### Add this for memory usage
perf_type_map = {'LoanID' : 'int64', 'Servicer' : 'category', 'CurrInterestRate' : 'float32', 'CAUPB' : 'float32',
                 'LoanAge' : 'int64', 'MonthsToMaturity' : 'int64', 'AdMonthsToMaturity' : 'float32', 'MSA' : 'int64',
                 'CLDS' : 'category', 'ModFlag' : 'category', 'ZeroBalCode' : 'float32', 'ZeroBalDate' : 'category',
                 'LastInstallDate' : 'category', 'ForeclosureCosts' : 'float32', 'PPRC' : 'float32',
                 'AssetRecCost' : 'float32', 'MHRC' : 'float32', 'ATFHP' : 'float32', 'NetSaleProceeds' : 'float32',
                 'CreditEnhProceeds' : 'float32', 'RPMWP' : 'float32', 'OFP' : 'float32', 'NIBUPB' : 'float32',
                 'PFUPB' : 'float32', 'RMWPF' : 'category', 'FPWA' : 'float32', 'ServicingIndicator' : 'category'}

aquisition_frame = pd.read_csv('Acquisition_2016Q4.txt', sep='|', names=col_acq, index_col=False, nrows=lines_to_read)
performance_frame = pd.read_csv('Performance_2016Q4.txt', sep='|', names=col_per, usecols = col_per_subset,
                                index_col=False, dtype=perf_type_map, nrows=lines_to_read)



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""" Calculate realised LGD """""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

realised LGD: LGD = (EAD - Discounted Recovery + costs) / EAD

# Variables of Interest
    ## (discounted) Recoveries:
    - NetSaleProceeds
    - OFP: Other Foreclosure Proceeds
    - FPWA: Foreclosure Principal Write-off Amount
    - CreditEnhProceeds: Credit Enhancement Proceeds
    - MortInsPerc: Mortgage Insurance Percentage
    - MortInsType: Mortgage Insurance Type
    - RPMWP: Repurchase Make Whole Proceeds
    - DispositionDate
    - CurrInterestRate

    ## Exposure at default (EaD):
    - CAUPB: Current Actual Unpaid Principal Balance

    ## Costs:
    - ForeclosureCosts
    - MHRC: Miscellaneous Holding Expenses and Credits
    - AssetRecCost: Asset Recovery Costs
    - PPRC: Property Preservation and Repair Costs
    - ATFHP: Associated Taxes for Holding Property

"""

# LGD: Loss Given Defualt -> select the defaulted loans.
obs_defaults = performance_frame[
    (performance_frame.CLDS != '0') &
    (performance_frame.CLDS != '1') &
    (performance_frame.CLDS != '2') &
    (performance_frame.CLDS != 'X')
    ][['LoanID', 'MonthRep', 'CAUPB']].drop_duplicates("LoanID") # LoanID, date of default and Exposure at default date
obs_defaults.columns = ['LoanID', 'DefaultDate', 'CAUPB']

# Select variables of interest
lgd_variables = ['NetSaleProceeds', 'OFP', 'FPWA', 'CreditEnhProceeds', 'RPMWP', 'DispositionDate', 'CurrInterestRate',
                 'ForeclosureCosts', 'ForeclosureDate','MHRC', 'AssetRecCost', 'PPRC', 'ATFHP']
lgd_inputs = pd.DataFrame()
for varN in lgd_variables:
    lgd_inputs[[varN, varN + '_MonthRep']] = performance_frame[~performance_frame[varN].isna()][[varN, "MonthRep"]]
lgd_inputs['LoanID'] = performance_frame.LoanID
lgd_inputs['MortInsPerc'] = aquisition_frame.MortInsPerc
lgd_inputs['MortInsType'] = aquisition_frame.MortInsType

# Merge variables of interest and defautled loan df
default_df = pd.merge(obs_defaults, lgd_inputs, on='LoanID', how='outer').dropna(subset=['CAUPB']) # Drop rows with missing EaD
default_df[default_df.select_dtypes(exclude=['category','datetime64[ns]']).columns] = default_df.select_dtypes(
    exclude=['category','datetime64[ns]']).fillna(0)
default_df['Default'] = 1
#default_df.CAUPB = (1 - default_df.MortInsPerc) * default_df.CAUPB # Mortgage Insurance
default_df.columns = [w.replace('_MonthRep', '_Date') for w in default_df.columns]

# Discounted Recoveries
def relTimer(end_date, start_date): return (end_date - start_date).apply(lambda d : d.days / 365 ).fillna(0)
default_df['recoveries'] = default_df.NetSaleProceeds * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.NetSaleProceeds_Date , default_df.DefaultDate))) \
                       + default_df.OFP * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.OFP_Date , default_df.DefaultDate))) \
                       + default_df.CreditEnhProceeds * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.CreditEnhProceeds_Date , default_df.DefaultDate))) \
                       + default_df.FPWA * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.FPWA_Date , default_df.DefaultDate))) \
                       + default_df.RPMWP * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.RPMWP_Date , default_df.DefaultDate)))
# Discounted costs
default_df['costs'] = default_df.ForeclosureCosts * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.ForeclosureCosts_Date , default_df.DefaultDate))) \
                  + default_df.MHRC * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.MHRC_Date, default_df.DefaultDate))) \
                  + default_df.PPRC * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.PPRC_Date, default_df.DefaultDate))) \
                  + default_df.AssetRecCost * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.CreditEnhProceeds_Date, default_df.DefaultDate))) \
                  + default_df.ATFHP * ((1 + default_df.CurrInterestRate)**(- relTimer(default_df.ATFHP_Date, default_df.DefaultDate)))
default_df['realised_LGD'] = ((default_df.CAUPB -default_df.recoveries + default_df.costs) / default_df.CAUPB)
default_df['realised_LGD'] = default_df['realised_LGD'].replace(1, 0.999)
default_df['realised_LGD'] = default_df['realised_LGD'].replace(0, 0.001)

# Aggregate with performance dataset
dfs = [performance_frame,
       aquisition_frame[['LoanID', 'OrLTV', 'OrCLTV', 'CreditScore', 'LoanPurpose', 'ProductType', 'CoCreditScore']],
       default_df[['LoanID','DefaultDate', 'Default', 'realised_LGD', 'recoveries', 'costs']]]
agg_df = reduce( lambda left, right: pd.merge(left, right, on='LoanID', how='outer').drop_duplicates('LoanID', keep='last'), dfs)
agg_df.Default = agg_df.Default.fillna(0)
#agg_df.realised_LGD = agg_df.realised_LGD.fillna(0)

# Split the data into development/validation sets
cutoff_date = datetime.date(2017, 12, 31)
development_set = agg_df[((agg_df.DefaultDate < cutoff_date) | (agg_df.DefaultDate.isna()))]
validation_set = agg_df[(agg_df.DefaultDate > cutoff_date) | (agg_df.DefaultDate.isna())]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""" Modelling LGD """""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from scipy.stats import norm
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

# Remove outlier with more than 80% missing values
development_set = pd.merge(development_set[['LoanID','realised_LGD']], development_set[development_set.columns[development_set.isnull().mean() < 0.8]], how='outer')

# Logistic transf
development_set['rLGD_logistic'] = np.log( development_set.realised_LGD / (1 - development_set.realised_LGD) )
development_set['rLGD_probit'] = norm.ppf( development_set.realised_LGD )

# Scale variables
min_max_scaler = MinMaxScaler()
int_flt_cols = development_set.select_dtypes(exclude=['category', 'datetime64[ns]']).columns
development_set[int_flt_cols] = min_max_scaler.fit_transform(development_set.select_dtypes(exclude=['category', 'datetime64[ns]']))

# Feature variables
FEATURES = ['CAUPB', 'OrLTV', 'CreditScore', 'CoCreditScore']
LABEL = 'rLGD_logistic' #realised_LGD, rLGD_probit
Y = development_set[LABEL]
X = development_set[FEATURES]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# Model
lregr = LinearRegression()
lregr.fit(X_train, y_train)
lregr_pred = lregr.predict(X_test)




