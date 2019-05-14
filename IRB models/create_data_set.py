import datetime
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import random
from sklearn   import preprocessing
from model     import *

class data_model(object):
     
     def catVAR2num(self, t, var, var_agr):
          ###Transform cathegorical variables to DRs###
          t=pd.merge(t, t.groupby([var], as_index=False)[var_agr].mean().rename(columns={var_agr:var+"_num"}), on=var, how='inner')
          return t
     
     def ScaleNUM(self, t, var):
          ###Scale numerical values between 0 and 1###
          x                   = t[var]
          x                   = x.fillna(0)
          x                   = x.replace(np.inf, 0)
          x                   = np.array(x.values).reshape(-1,1) 
          min_max_scaler      = preprocessing.MinMaxScaler()
          x_scaled            = min_max_scaler.fit_transform(x)
          t[var+'_scaled']    = pd.DataFrame(x_scaled)
          return
        
     def prepare_data(self, df, ldate):
          ###Convert dates to datetime###
          df                    = df.assign(obs_dt=pd.to_datetime(df.issue_d, format="%b-%Y"))
          ###Removing missing default date (use next_payment_date and last_payment_date to infer default_date)###
          df["Default_date"]    = df.last_pymnt_d.fillna(df.next_pymnt_d)
          df                    = df[pd.notnull(df.Default_date)]
          df                    = df.assign(Default_date=pd.to_datetime(df.Default_date, format="%b-%Y"))
          ####Default definition (given in the data model)###
          dflt_definition       = ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
          df['Default_Binary']  = df.loan_status.apply(lambda s : 1 if s in dflt_definition else 0)
          ###Convert Rating grade into numberse###
          df['grade_num']       = df.grade.apply(lambda x: {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}[x])
          ###Transform categorical variables to numeric###
          df                    = self.catVAR2num(df, 'purpose', 'Default_Binary')
          df                    = self.catVAR2num(df, 'home_ownership', 'Default_Binary')
          df                    = self.catVAR2num(df, 'sub_grade', 'Default_Binary')
          df                    = self.catVAR2num(df, 'addr_state', 'Default_Binary')
          df                    = self.catVAR2num(df, 'emp_length', 'Default_Binary')
          ###Rescale numerical variables###
          self.ScaleNUM(df, 'int_rate')
          self.ScaleNUM(df, 'funded_amnt')
          self.ScaleNUM(df, 'inq_last_6mths')
          ###Ratio example###
          df['Income2TB']       = df['annual_inc'] / df['tot_cur_bal']
          self.ScaleNUM(df, 'Income2TB')
          ###Add realised LGD and CCF###
          df.term                                      = df.term.str.replace(" months", "").astype(dtype=np.float64)
          df["original_exposure"]                      = df.installment * df.term
          df["EAD_realised"]                           = df.original_exposure - df.total_pymnt  # Original amout - Amount already paid
          df["CCF_realised"]                           = np.maximum(0, 1 - pd.to_numeric(df['all_util'])/100)
          df.CCF_realised[df.all_util.isnull()]        = np.maximum(0, df.EAD_realised[df.all_util.isnull()] / (df.installment[df.all_util.isnull()] * df.term[df.all_util.isnull()]))
          end_date                                     = datetime.date(2016, 1, 1)
          time_in_default                              = end_date - df.Default_date.dt.date
          df["time_in_default"]                        = time_in_default.apply(lambda d: d.days / 365)
          df["LGD_realised"]                           = (df.EAD_realised + df.collection_recovery_fee - df.recoveries * (1 + df.int_rate/100) ** (-df.time_in_default)) / (df.EAD_realised + df.collection_recovery_fee)
          df["LGD_realised"]                           = np.minimum(1, np.maximum(0, df.LGD_realised) )
          df.LGD_realised[df.Default_Binary == 0]      = float('NaN')
          df.CCF_realised[df.Default_Binary == 0]      = float('NaN')
          df['Bin_CCF_realised']                       = float('NaN')
          df['Bin_LGD_realised']                       = float('NaN')
          df.EAD_realised[df.Default_Binary == 0]      = float('NaN')
          df.LGD_realised[df.Default_Binary == 1].fillna(0)
          df.CCF_realised[df.Default_Binary == 1].fillna(0)
          df.EAD_realised[df.Default_Binary == 1].fillna(0)
          ###Define development period and validation period###
          development_set = df[(df.obs_dt.dt.date < ldate)] # Application before ldate
          validation_set = df[(df.obs_dt.dt.date > ldate) | (df.Default_date.dt.date > ldate)] #Application after ldate OR default after ldate
          ###Add PD variable###
          FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num', 'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
          LABEL = 'Default_Binary'
          development_set, validation_set         = model().PD_model(FEATURES, LABEL, development_set, validation_set, 'PD')
          ###LGD model###
          FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num', 'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
          LABEL = 'LGD_realised'
          development_set, validation_set         = model().LGD_model(FEATURES, LABEL, development_set, validation_set, 'LGD')
          ###CCF model###
          FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num', 'funded_amnt_scaled', 'int_rate_scaled', 'inq_last_6mths_scaled', 'Income2TB_scaled']
          LABEL = 'CCF_realised'
          development_set, validation_set         = model().CCF_model(FEATURES, LABEL, development_set, validation_set, 'CCF')
          FEATURES = ['home_ownership_num', 'purpose_num', 'addr_state_num', 'emp_length_num']
          LABEL = 'CCF_realised'
          development_set, validation_set         = model().CCF_model(FEATURES, LABEL, development_set, validation_set, 'CCF_')
          ###Bin PDs###
          development_set, validation_set         = binning().binning_monotonic(development_set, validation_set, 'PD', 'Default_Binary', 75, 7)
          ###Bin CCF###
          development_set, validation_set         = binning().binning_monotonic(development_set, validation_set, 'CCF',  'CCF_realised', 75, 7)
          development_set['Bin_CCF_realised'][development_set.CCF_realised != float('NaN')]    = binning().binning_std(development_set.CCF_realised[development_set.CCF_realised != float('NaN')])
          validation_set['Bin_CCF_realised'][validation_set.CCF_realised != float('NaN')]      = binning().binning_std(validation_set.CCF_realised[validation_set.CCF_realised != float('NaN')])
          development_set, validation_set         = binning().binning_monotonic(development_set, validation_set, 'CCF_', 'CCF_realised', 65, 7)
          ### Bin LGD###
          development_set, validation_set         = binning().binning_monotonic(development_set, validation_set, 'LGD', 'LGD_realised', 75, 7)
          development_set['Bin_LGD_realised'][development_set.LGD_realised != float('NaN')]    = binning().binning_std(development_set.LGD_realised[development_set.LGD_realised != float('NaN')])
          validation_set['Bin_LGD_realised'][validation_set.LGD_realised != float('NaN')]      = binning().binning_std(validation_set.LGD_realised[validation_set.LGD_realised != float('NaN')])
          return development_set, validation_set

class import_data(object):
     
     def EY(self, link):
          ### Create development and validation data set ###
          # Cross check with 2.5.1: Specific definitions. 
          ldate = datetime.date(2015, 1, 1)
          data = pd.read_csv(link + "/loan.csv", low_memory=False)
          development_set, validation_set = data_model().prepare_data(data, ldate)
          return development_set, validation_set