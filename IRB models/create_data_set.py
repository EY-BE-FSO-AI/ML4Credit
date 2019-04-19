import pandas as pd
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

class data_model(object):

    def __init__(self, data, ldate):
        """
        Create data helper object
        :param data: pandas dataframe (see data model)
        :param ldate: limit date between development / monitoring sets
        """
        self.data = data
        self.ldate = ldate

    def split_data(self):
        """
        :return: Tuple - (development set, monitoring sets)
        """
        df = self.data.copy()
        # Convert dates to datetime
        df['issue_dt'] = df.issue_d.apply(lambda d: datetime.datetime.strptime(d, "%b-%Y").date())

        #Removing missing default date (use next_payment_date and last_payment_date to infer default_date)
        df["Default_date"] = df.last_pymnt_d.fillna(df.next_pymnt_d)
        df = df[pd.notnull(df.Default_date)]
        df["Default_date"] = df.Default_date.apply(lambda d: datetime.datetime.strptime(d, "%b-%Y").date())

        #Default definition (given in the data model)
        dflt_definition = ['Default', 'Charged Off', 'Late (31-120 days)', 'Late (16-30 days)', 'Does not meet the credit policy. Status:Charged Off']
        df['Default_Binary'] = df.loan_status.apply(lambda s : 1 if s in dflt_definition else 0)

        ###Transform cathegorical variables to DRs###
        def catVAR2num(t, var, var_agr):
          t=pd.merge(t, t.groupby([var], as_index=False)[var_agr].mean().rename(columns={var_agr:var+"_num"}), on=var, how='inner')
          return t
               
        df = catVAR2num(df, 'purpose', 'Default_Binary')
        df = catVAR2num(df, 'home_ownership', 'Default_Binary')
        df = catVAR2num(df, 'grade', 'Default_Binary')
        df = catVAR2num(df, 'sub_grade', 'Default_Binary')
        df = catVAR2num(df, 'addr_state', 'Default_Binary')
        df = catVAR2num(df, 'emp_length', 'Default_Binary')
          
        ###Scale numerical values between 0 and 1###
        def ScaleNUM(t, var):
             x = t[var]
             x = x.fillna(0)
             x = x.replace(np.inf, 0)
             x = np.array(x.values).reshape(-1,1) 
             min_max_scaler = preprocessing.MinMaxScaler()
             x_scaled = min_max_scaler.fit_transform(x)
             t[var+'_scaled'] = pd.DataFrame(x_scaled)
             return
          
        ScaleNUM(df, 'int_rate')
        ScaleNUM(df, 'funded_amnt')
        ScaleNUM(df, 'inq_last_6mths')
        
        ###Ratio example###
        df['Income2TB'] = df['annual_inc'] / df['tot_cur_bal']
        ScaleNUM(df, 'Income2TB')

        # Define development period and monitoring period
        df_dev = df[(df.issue_dt < self.ldate)] # Application before ldate
        df_monit = df[(df.issue_dt > self.ldate) | (df.Default_date > self.ldate)] #Application after ldate OR default after ldate

        return df_dev, df_monit



if __name__ == '__main__':
    df = pd.read_csv(os.path.normpath(os.path.expanduser("~/Documents/Python")) + "/loan.csv", low_memory=False)
    development_set, monitoring_set = data_model(data = df, ldate= datetime.date(2015, 1, 1)).split_data()
    #Plot application years for dev set:
    development_set.issue_dt.apply(lambda d: d.year).value_counts().plot.bar()
    plt.show()