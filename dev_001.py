### Import Packages
######################################################################################################
import collections
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.impute import SimpleImputer
import random
import time
import xgboost as xgb


### File Paths
######################################################################################################
config_folder_path = 'D:/kemper_exercise/raw_data/'
config_train_file_name = 'retention_sample.csv'


### Column Configuration
######################################################################################################

config_y_col = 'RETAINED'
config_id_col = 'FAKE_POLICY_NUMBERS'
config_categ_x_cols = ['PDW_PACKAGE_POLICY_FLAG', 'RATE_STATE_NAME', 'TIER_CODE',
                        'NEW_RENEWAL_CODE', 'OCCUPATION']
config_contin_x_cols = ['IBS_CREDIT_SCORE_NUMBER', 'ADDRESS_YEARS', 'EMPLOYED_YEARS',
                        'POLICY_TERM_PREMIUM_AMOUNT', 'POLICY_TERM_NUMBER_OF_MONTHS',
                        '0_3_MAJOR', '3_5_MAJOR', '0_3_MINOR', '3_5_MINOR', 'OPERATOR_AGE', 'VEH_ISO_LIAB_SYM']

config_x_cols = config_categ_x_cols + config_contin_x_cols


### Define Functions
######################################################################################################



class MissingnessIndicatorTransformer(BaseEstimator, TransformerMixin):
  """
  Replace missing values in continuous fields with zero and create missingness
  indicators in new columns. e.g. columns 'A_Missing', 'B_Missing' are
  returned in addition to 'A' and 'B'
  """
  
  @staticmethod
  def binary_missingness(iterable):
      return [1 if (np.isnan(x) or np.isinf(x)) else 0 for x in iterable]
      
  def fit(self, target):
    return self

  def transform(self, target):
    target_copy = target.copy()
    target_missing_ind_list = []
    for c in target_copy.columns:
        col_c_indicator = pd.DataFrame({f'{c}_Missing' : self.binary_missingness(iterable = target_copy[c])})
        target_missing_ind_list.append(col_c_indicator)
    target_missing_ind = pd.concat(target_missing_ind_list, axis = 1)
    output_copy = pd.concat([target_copy.fillna(0).reset_index(drop=True), target_missing_ind.reset_index(drop=True)], axis = 1)
    print(f'output shape: {output_copy.shape[0]})')
    
    return output_copy



class CategoricalTransformer(BaseEstimator, TransformerMixin):
  """
  One-hot encode categorical columns in a way that will work when
  the test set has values not found in the training set. Columns
  returned will exclude a single reference level. e.g.
  if a pandas.DataFrame with column 'X' and values 'A', 'B', and 'C'
  is passed where frequencies of 'A', 'B', and 'C' and 500, 250, and 250,
  respectively, binary columns 'X_B' and 'X_C' will be returned.
  """
  def __init__(self, column_value_counts = {}):
      self.column_value_counts = column_value_counts
      
  def fit(self, target):
      for column in target.columns:
          value_count_series = target[column].value_counts()
          key_values = list(value_count_series.index)
          key_counts = list(value_count_series.values)
          self.column_value_counts[column] = dict(zip(key_values, key_counts))
      return self

  def transform(self, target):
      target_copy = target.copy()
      one_hot_cols = list(set(self.column_value_counts.keys()))
      one_hot_df_list = []
      for ohc in one_hot_cols:
          ohc_dict = self.column_value_counts.get(ohc)
          
          # Remove reference level (max frequency categorical level)
          ohc_keys = list(ohc_dict.keys())
          ohc_values = list(ohc_dict.values())
          use_levels = [ohc_keys for _, ohc_keys in sorted(zip(ohc_values, ohc_keys))][:-1]
          for level in use_levels:
              binary_values = [1 if x == level else 0 for x in target_copy[ohc]]
              encoded_values = pd.DataFrame({f'{ohc}_{level}' : binary_values})
              one_hot_df_list.append(encoded_values)
      target_copy = pd.concat(one_hot_df_list, axis = 1)
      return target_copy



class LowVarianceTransformer(BaseEstimator, TransformerMixin):
  """
  Removes columns in numpy array with zero variance based on the training set.
  This is necessary after missingness indicators are created for
  every column - including columns without missing values.
  """
  def __init__(self, zero_variance_cols = []):
      self.zero_variance_cols = zero_variance_cols
      
  def fit(self, target):
      self.zero_variance_cols = [i for i in list(range(target.shape[1])) if len(np.unique(target[:,i])) == 1]
      #self.zero_variance_cols = [i for c in target.columns if len(np.unique(target[c])) == 1]
      return self

  def transform(self, target):
      target_copy = target.copy()
      keep_cols = [c for c in list(range(target.shape[1])) if c not in self.zero_variance_cols]
      target_copy = target_copy[:, keep_cols]
      return target_copy






### Execute
######################################################################################################

df = pd.read_csv(f'{config_folder_path}{config_train_file_name}')
x = df[config_x_cols]
y = df[config_y_col]


# Define Pipeline
numeric_transformer = Pipeline(steps = [('missingness', MissingnessIndicatorTransformer()),
                                        ('scaler', StandardScaler()),
                                        ('zero variance column removal', LowVarianceTransformer())])
    

categorical_transformer = Pipeline(steps = [('custom categorical encoder', CategoricalTransformer())])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, config_contin_x_cols),
                                               ('cat', categorical_transformer, config_categ_x_cols)], remainder = 'passthrough')
    

    
pipeline = Pipeline(steps = [('preprocessor', preprocessor)])




train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(x, y, test_size = 0.2, random_state = 912)

#pipeline.fit(train_x)
train_x = pipeline.fit_transform(train_x)
test_x = pipeline.transform(test_x)








### TO DO
######################################################################################################
# transformer for response variable
# combined transformer/pipeline object
# start xgbtuner





































