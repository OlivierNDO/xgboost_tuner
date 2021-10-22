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
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, MinMaxScaler
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
def create_partitions_by_id(pandas_dframe : pd.DataFrame, id_column = None,
                            kfold_colname = 'k_fold', k = 10, train_size = 0.7,
                            random_seed = 8292021):
    """
    Create train, test, and validation partitions ensuring that identifiers
    (like customer number) only appear in a single partition. The 'train' DataFrame
    returned will have a k-fold column added.
    Args:
        pandas_dframe: pandas.DataFrame from which to create partitions
        id_column: column name with identifier field
        kfold_colname: column name to add in train that has the fold integer
        k: number of cross validation partitions to include in the training set
        train_size: percentage of dataset to keep in training set
        random_seed: integer to use in random number generator & shuffling
    Returns:
        train, test, validation (pandas.DataFrame objects)
    """
    # Copy DataFrame & Create Train, Test, Validation Identifier Lists
    df_copy = pandas_dframe.copy()
    random.seed(random_seed)
    if id_column is not None:
        unique_ids = list(set(df_copy[id_column]))
    else:
        df_copy['id_column'] = list(range(df_copy.shape[0]))
        id_column = 'id_column'
        unique_ids = list(set(df_copy[id_column]))
    random.shuffle(unique_ids)    
    train_ids = random.sample(unique_ids, int(len(unique_ids) * train_size))
    test_valid_ids = [ui for ui in unique_ids if ui not in train_ids]
    test_ids = random.sample(test_valid_ids, int(0.5 * len(test_valid_ids)))
    valid_ids = [z for z in test_valid_ids if z not in test_ids]
    
    # Create Train, Test, Validation DataFrames
    train_df = df_copy[df_copy[id_column].isin(train_ids)]
    test_df = df_copy[df_copy[id_column].isin(test_ids)]
    valid_df = df_copy[df_copy[id_column].isin(valid_ids)]
    
    # Add K-Fold Column to Train DataFrame
    kfold_list = list(range(1, (k + 1))) * int(np.ceil(train_df.shape[0] / k))
    random.shuffle(kfold_list)
    train_df[kfold_colname] = kfold_list[:train_df.shape[0]]
    print(f"Adding column '{kfold_colname}' to training dataframe with {k} folds")
    n_train, n_test, n_valid = train_df.shape[0], test_df.shape[0], valid_df.shape[0]
    print(f"Returning train (n = {n_train}), test (n = {n_test}), and validation (n = {n_valid}) sets")
    return train_df, test_df, valid_df



### Execute
######################################################################################################
# Read Data
df = pd.read_csv(f'{config.folder_path}{config.train_file_name}')


# Split into Test & Train
train, test = sklearn.model_selection.train_test_split(df, test_size = 0.2, random_state = 912)
test, valid = sklearn.model_selection.train_test_split(test, test_size = 0.5, random_state = 912)


# Transform Predictor Features
feature_pipeline = trans_mod.FeaturePipeline(train_df = train,
                                             test_df = test,
                                             valid_df = valid,
                                             numeric_columns = config.contin_x_cols,
                                             categorical_columns = config.categ_x_cols,
                                             numeric_transformers = [trans_mod.MissingnessIndicatorTransformer(),
                                                                     trans_mod.ZeroVarianceTransformer(),
                                                                     trans_mod.InteractionTransformer(interaction_list = config.interaction_cols),
                                                                     trans_mod.PolynomialTransformer(feature_power_dict =  config.polynomial_col_dict),
                                                                     trans_mod.CustomScalerTransformer()],
                                             categorical_transformers = [trans_mod.CategoricalTransformer(),
                                                                         trans_mod.ZeroVarianceTransformer()],
                                             pipeline_save_path = config.feature_pipeline_save_path)

train_x, test_x, valid_x  = feature_pipeline.process_train_test_valid_features()
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

"""
# add process_train_test_valid_response
# response pipeline for continuous response variable?
# start xgbtuner
# write unit tests
# Work on docstrings


"""
















