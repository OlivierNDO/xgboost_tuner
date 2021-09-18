### Import Packages
######################################################################################################
import collections
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.impute import SimpleImputer
import sys
import random
import time
import xgboost as xgb



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


### File Paths
######################################################################################################
config_folder_path = 'D:/kemper_exercise/raw_data/'
config_train_file_name = 'retention_sample.csv'



### Filename & Directory Configuration
######################################################################################################
config_parent_directory = 'D:/xgboost_tuner/'
config_model_save_dir = f'{config_parent_directory}saved_models/'
config_pipeline_save_path = f'{config_model_save_dir}transformation_pipeline.pkl'


### Import Modules
######################################################################################################
sys.path.append(config_parent_directory)
import tuner.utils.transformation_module as trans_mod




### Define Functions
######################################################################################################





### Execute
######################################################################################################

# Read Data
df = pd.read_csv(f'{config_folder_path}{config_train_file_name}')
x = df[config_x_cols]
y = df[config_y_col]


# Split into Test & Train
train, test = sklearn.model_selection.train_test_split(df, test_size = 0.2, random_state = 912)


transformer = trans_mod.FeatureTransformer(train_df = train,
                                           test_df = test,
                                           numeric_columns = config_contin_x_cols,
                                           categorical_columns = config_categ_x_cols,
                                           pipeline_save_path = config_pipeline_save_path)


train_x, test_x  = transformer.process_train_test_features()

transformer.save_pipeline()








### TO DO
######################################################################################################
# transformer for response variable
# single pipeline class
# serialize pipeline (https://stackoverflow.com/questions/57888291/how-to-properly-pickle-sklearn-pipeline-when-using-custom-transformer)
# create pipeline module
# create configuration script
# start xgbtuner
# write unit tests
# Work on docstrings

