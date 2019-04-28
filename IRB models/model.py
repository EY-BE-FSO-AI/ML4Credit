###Owner code: Brent Oeyen
###Comments: 
###          improve speed: look at join on intervals currently implemented using numpy
########################################################################################################################

import pandas            as pd
import numpy             as np
import itertools
import tensorflow        as tf
import scipy.stats.stats as stats
from tensorflow.contrib.learn.python.learn             import metric_spec
from tensorflow.contrib.learn.python.learn.estimators  import _sklearn
from tensorflow.contrib.learn.python.learn.estimators  import estimator
from tensorflow.contrib.learn.python.learn.estimators  import model_fn
from tensorflow.python.framework                       import ops
from tensorflow.python.saved_model                     import loader
from tensorflow.python.saved_model                     import tag_constants
from tensorflow.python.util                            import compat

class model(object):

     def PD_model(self, FEATURES, LABEL, development, monitoring, var_name):
     
          def input_fn(data_set):
              feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} 
              labels = tf.constant(data_set[LABEL].values)
              return feature_cols, labels
              
          feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
          
          regressor = tf.contrib.learn.DNNRegressor(
            feature_columns=feature_cols, hidden_units=[64, 32, 16], optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.001) )
          regressor.fit(input_fn=lambda: input_fn(development), steps=56)
          
          y1 = regressor.predict(input_fn=lambda: input_fn(development))
          y2 = regressor.predict(input_fn=lambda: input_fn(monitoring))
          development[var_name] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1])))).values
          monitoring [var_name] = pd.DataFrame(list(itertools.islice(y2, len(monitoring .iloc[:,1])))).values
          
          return development, monitoring
      
     def LGD_model(self, FEATURES, LABEL, development, monitoring, var_name):

          def input_fn(data_set):
              feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} 
              labels = tf.constant(data_set[LABEL].values)
              return feature_cols, labels
              
          feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
          
          regressor = tf.contrib.learn.DNNRegressor(
            feature_columns=feature_cols, hidden_units=[64, 32, 16], optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.001) )
          regressor.fit(input_fn=lambda: input_fn(development[development.Default_Binary == 1]), steps=56)
          
          y1 = regressor.predict(input_fn=lambda: input_fn(development))
          y2 = regressor.predict(input_fn=lambda: input_fn(monitoring))
          development[var_name] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1])))).values
          monitoring [var_name] = pd.DataFrame(list(itertools.islice(y2, len(monitoring .iloc[:,1])))).values
          
          return development, monitoring
          
     def CCF_model(self, FEATURES, LABEL, development, monitoring, var_name):

          def input_fn(data_set):
              feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES} 
              labels = tf.constant(data_set[LABEL].values)
              return feature_cols, labels
              
          feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]
          
          regressor = tf.contrib.learn.DNNRegressor(
            feature_columns=feature_cols, hidden_units=[64, 32, 16], optimizer=tf.train.ProximalAdagradOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=0.001) )
          regressor.fit(input_fn=lambda: input_fn(development[development.Default_Binary == 1]), steps=56)
          
          y1 = regressor.predict(input_fn=lambda: input_fn(development))
          y2 = regressor.predict(input_fn=lambda: input_fn(monitoring))
          development[var_name] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1])))).values
          monitoring [var_name] = pd.DataFrame(list(itertools.islice(y2, len(monitoring .iloc[:,1])))).values
          
          return development, monitoring

     def binning_monotonic(self, development, monitoring, var, var2, quantiles):
          development["Quantiles"]      = pd.to_numeric(pd.qcut(development[var], q = quantiles, labels=np.arange(0, quantiles))) #Calculate quantiles
          t1                            = pd.merge(    development[['Quantiles', var ]].astype(float).groupby('Quantiles', as_index=False).agg({var: [np.min, np.max]}), \
                                                       development[['Quantiles', var2]].astype(float).groupby('Quantiles', as_index=False).agg({var2: [np.mean]}) \
                                                       , on='Quantiles', how='inner') #Calculate mean of the realised values for a given interval of modelled values
          t1.columns                    = ["Quantiles", "MIN", "MAX", "Exp"]
          development.drop(['Quantiles'], axis=1, inplace=True)
          def mono_bin(x, a, b, n, var):
               Y = x[a]
               X = x[b]
               r = 0
               while np.abs(r) < 1:  #Continue reducing the number of bins until the outcome is monotone
                    t1        = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})  #Split values into uniform observations
                    t2        = t1.groupby('Bucket', as_index = True)
                    r, p      = stats.spearmanr(t2.mean().X, t2.mean().Y)
                    n         = n - 1
               t3                   = t2.min().X
               t3                   = t3.to_frame()
               t3.columns           = ['min_' + b]
               t3['max_' + b]       = t2.max().X
               t4                   = (t3.sort_index(by = 'min_' + b)).reset_index(drop = True)
               t4['Bin_' + var]     = t4.index + 1
               i, j                 = np.where((x[b][:, None] >= t4['min_' + b].values) & (x[b][:, None] <= t4['max_' + b].values))
               t5                   = pd.DataFrame(np.column_stack([x.values[i], t4.values[j]]), columns=x.columns.append(t4.columns))
               t5.drop(['Quantiles', 'min_' + b, 'max_' + b, 'Exp'], axis=1, inplace=True)
               return t5

          t2 = mono_bin(t1, 'Exp', 'Quantiles', 7, var)

          i, j      =    np.where((development[var][:, None] >= t2.MIN.values) & (development[var][:, None] <= t2.MAX.values))
          t3        =    pd.DataFrame(np.column_stack([development.values[i], t2.values[j]]),columns=development.columns.append(t2.columns))
          t3.drop(['MIN', 'MAX'], axis=1, inplace=True)
          i, j      =    np.where((monitoring[var][:, None] >= t2.MIN.values) & (monitoring[var][:, None] <= t2.MAX.values))
          t4        =    pd.DataFrame(np.column_stack([monitoring.values[i], t2.values[j]]),columns=monitoring.columns.append(t2.columns))
          t4.drop(['MIN', 'MAX'], axis=1, inplace=True)
          return t3, t4