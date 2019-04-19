import pandas as pd
import itertools
import tensorflow as tf
from tensorflow.contrib.learn.python.learn import metric_spec
from tensorflow.contrib.learn.python.learn.estimators import _sklearn
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import model_fn
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.util import compat

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
          development[var_name] = pd.DataFrame(list(itertools.islice(y1, len(development.iloc[:,1]))))
          monitoring [var_name] = pd.DataFrame(list(itertools.islice(y2, len(monitoring .iloc[:,1]))))
          
          return development, monitoring