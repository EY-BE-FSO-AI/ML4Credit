###Owner code: Brent Oeyen
###Comments: 
###          
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

     def PD_model(self, FEATURES, LABEL, development, validation, var):
     
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
          y2 = regressor.predict(input_fn=lambda: input_fn(validation))
          development[var] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1])))).values
          validation [var] = pd.DataFrame(list(itertools.islice(y2, len(validation .iloc[:,1])))).values
          
          return development, validation
      
     def LGD_model(self, FEATURES, LABEL, development, validation, var):

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
          y2 = regressor.predict(input_fn=lambda: input_fn(validation))
          development[var] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1])))).values
          validation [var] = pd.DataFrame(list(itertools.islice(y2, len(validation .iloc[:,1])))).values
          
          return development, validation
          
     def CCF_model(self, FEATURES, LABEL, development, validation, var):

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
          y2 = regressor.predict(input_fn=lambda: input_fn(validation))
          development[var] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1])))).values
          validation [var] = pd.DataFrame(list(itertools.islice(y2, len(validation .iloc[:,1])))).values
          
          return development, validation

class binning(object):
     
     def match_interval(self, x, bins=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], out=range(1,13)):
          z = [out[i] for i in np.searchsorted(bins, x)]
          return z
     
     def mono_bin(self, x, a, b, n, var):
          Y = x[a]
          X = x[b]
          r = 0
          while np.abs(r) < 1:  #Continue reducing the number of bins until the outcome is monotone
               t1        = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n)})  #Split values into uniform observations
               t2        = t1.groupby('Bucket', as_index = True)
               r, p      = stats.spearmanr(t2.mean().X, t2.mean().Y)
               n         = n - 1
          t3                   = t2.max().X
          t3                   = t3.to_frame()
          t3.columns           = ['max_' + b]
          t4                   = x
          t4['Bin_' + var]     = self.match_interval(X, t3['max_' + b].values, range(1, t3['max_' + b].count()+2))
          t4.drop(['Exp'], axis=1, inplace=True)
          return t4

     def binning_monotonic(self, development, validation, var, var2, quantiles, n):
          #var: Variable used to calcualte the quantiles
          #var2: Variable used to evaluate the ranking per quantile
          #quantiles: number of quantiles
          development["Quantiles"]      = pd.to_numeric(pd.qcut(development[var], q = quantiles, labels=np.arange(0, quantiles))) #Calculate quantiles
          t1                            = pd.merge(    development[['Quantiles', var ]].astype(float).groupby('Quantiles', as_index=False).agg({var: [np.max]}), \
                                                       development[['Quantiles', var2]].astype(float).groupby('Quantiles', as_index=False).agg({var2: [np.mean]}) \
                                                       , on='Quantiles', how='inner') #Calculate mean of the realised values for a given interval of modelled values
          t1.columns                    = ["Quantiles", "MAX", "Exp"]
          development.drop(['Quantiles'], axis=1)
          t2                                      = self.mono_bin(t1, 'Exp', 'Quantiles', n, var)
          bins                                   = [0]*(t2['Bin_' + var].count()+2)
          bins[0:(t2['Bin_' + var].count()+1)]   = t2['Bin_' + var].values
          bins[-1]                               = t2['Bin_' + var].iloc[-1]
          t3                                      = development
          t3['Bin_' + var]                        = self.match_interval(development[var].values, t2.MAX, bins)
          t4                                      = validation
          t4['Bin_' + var]                        = self.match_interval(validation [var].values, t2.MAX, bins)
          return t3, t4
          
     def binning_std(self, data):
          y = self.match_interval(data*100)
          return y