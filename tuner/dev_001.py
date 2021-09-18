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


transformer = trans_mod.FeatureTransformer(train_df = train,
                                           test_df = test,
                                           numeric_columns = config.contin_x_cols,
                                           categorical_columns = config.categ_x_cols,
                                           pipeline_save_path = config.pipeline_save_path)


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

