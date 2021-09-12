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



class CustomTargetTransformer(BaseEstimator, TransformerMixin):
  # no need to implement __init__ in this particular case
  
  def fit(self, target):
    return self

  def transform(self, target):
    print('\n%%%%%%%%%%%%%%%custom_target_transform() called.\n')
    target_ = target.copy() 
    target_ = np.sqrt(target_)
    return target_

  # need to implement this too
  def inverse_transform(self, target):
    print('\n%%%%%%%%%%%%%%%custom_inverse_target_transform() called.\n')
    target_ = target.copy() 
    target_ = target_ ** 2
    return target_





def process_contin_features(self):
    # Create binary missingness indicators and replace nan, inf values with 0
    train_series, test_series, valid_series = [], [], []
    cols_with_missing_vals = 0
    for i, c in enumerate(self.contin_cols):
        na_inf_count = count_nan_inf_values(self.train_dframe, c) + \
        count_nan_inf_values(self.test_dframe, c) + \
        count_nan_inf_values(self.valid_dframe, c)
        if na_inf_count > 0:
            print(f'Replacing {na_inf_count} NaN/Inf values in {c} and creating missingness indicator {c}_Missing')
            train_series.append(pd.DataFrame({f'{c}_Missing' : [1 if (np.isnan(x) or np.isinf(x)) else 0 for x in self.train_dframe[c]]}))
            test_series.append(pd.DataFrame({f'{c}_Missing' : [1 if (np.isnan(x) or np.isinf(x)) else 0 for x in self.test_dframe[c]]}))
            valid_series.append(pd.DataFrame({f'{c}_Missing' : [1 if (np.isnan(x) or np.isinf(x)) else 0 for x in self.valid_dframe[c]]}))
            cols_with_missing_vals += 1
    print(f'Replaced NaN/Inf values in {cols_with_missing_vals} continuous features')
        







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
    #target_missing_ind = pd.concat(pd.DataFrame({f'{c}_Missing' : [binary_missingness(target_copy[c]) for c in target_copy.columns]}), axis = 1)
    target_copy = pd.concat([target_copy.fillna(0), target_missing_ind], axis = 1)
    return target_copy



class CategoricalTransformer(BaseEstimator, TransformerMixin):
  """
  Replace missing values in continuous fields with zero and create missingness
  indicators in new columns. e.g. columns 'A_Missing', 'B_Missing' are
  returned in addition to 'A' and 'B'
  """
  
  
      
  def fit(self, target):
    return self

  def transform(self, target):
    target_copy = target.copy()
    
    
    
    return target_copy







sklearn.base.BaseEstimator


### Execute
######################################################################################################

df = pd.read_csv(f'{config_folder_path}{config_train_file_name}')
x = df[config_x_cols]
y = df[config_y_col]



temp = MissingnessIndicatorTransformer()
temp.fit_transform(x[['IBS_CREDIT_SCORE_NUMBER']]).shape
temp.fit_transform(x[['ADDRESS_YEARS']]).shape





numeric_transformer = Pipeline(steps=[
    ('missingness', MissingnessIndicatorTransformer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, config_contin_x_cols)])#,
    #    ('cat', categorical_transformer, config_categ_x_cols)]
    #)
    
    
pipeline = Pipeline(steps = [('preprocessor', preprocessor)])



#catgorical_pipeline = Pipeline()

['IBS_CREDIT_SCORE_NUMBER', 'ADDRESS_YEARS', 'EMPLOYED_YEARS']

len([1 if (np.isnan(x) or np.isinf(x)) else 0 for x in train_x['IBS_CREDIT_SCORE_NUMBER']])


len([1 if (np.isnan(x) or np.isinf(x)) else 0 for x in train_x['ADDRESS_YEARS']])



train_x, train_y, test_x, test_y = sklearn.model_selection.train_test_split(x, y, test_size = 0.2, random_state = 912)


train_x = pipeline.fit_transform(train_x)



# https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf












git clone https://github.com/OlivierNDO/xgboost_tuner.git
















