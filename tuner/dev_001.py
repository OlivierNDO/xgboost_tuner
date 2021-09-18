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


### Import Modules
######################################################################################################
parent_directory = 'D:/xgboost_tuner/'
sys.path.append(parent_directory)
import tuner.utils.transformation_module as trans_mod
import tuner.utils.configuration as config


### Define Functions
######################################################################################################




### Execute
######################################################################################################
# Read Data
df = pd.read_csv(f'{config.folder_path}{config.train_file_name}')


# Split into Test & Train
train, test = sklearn.model_selection.train_test_split(df, test_size = 0.2, random_state = 912)

# Transform Predictor Features
feature_pipeline = trans_mod.FeaturePipeline(train_df = train,
                                                   test_df = test,
                                                   numeric_columns = config.contin_x_cols,
                                                   categorical_columns = config.categ_x_cols,
                                                   pipeline_save_path = config.feature_pipeline_save_path)

train_x, test_x  = feature_pipeline.process_train_test_features()
feature_pipeline.save_pipeline()


# Transform Response Variable
response_pipeline = trans_mod.ResponsePipeline(train_df = train,
                                               test_df = test,
                                               response_column = config.y_col,
                                               pipeline_save_path = config.response_pipeline_save_path)

train_y, test_y  = response_pipeline.process_train_test_response()
response_pipeline.save_pipeline()






### TO DO
######################################################################################################
# start xgbtuner
# write unit tests
# Work on docstrings

