### Import Packages
######################################################################################################
import collections
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from operator import itemgetter
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



### Pipeline Transformations
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
                                               valid_df = valid,
                                               response_column = config.y_col,
                                               pipeline_save_path = config.response_pipeline_save_path)

train_y, test_y, valid_y  = response_pipeline.process_train_test_valid_response()
response_pipeline.save_pipeline()


### XGBoost Hyperparameter Tuning with Early Stopping & Pipeline Transformations
######################################################################################################
# By using a pipeline transformation during  each of K folds, cross validation results most accurately measure true out of sample performance




class XgboostClassificationTuner:
    """
    Perform k fold cross validation over a random selection of
    hyperparameter space  using train, test, and validation sets.
    Each k-fold iteration assigns 1 fold to test, 1 fold to validation,
    and remaining folds to train. A custom sklearn Pipeline object
    is required to most accurately measure true out of sample performance
    """
    
    
    
    def __init__(self,
                 x : pd.DataFrame,
                 y : pd.DataFrame,
                 param_dict :  dict,
                 k_folds = 5,
                 n_boost_rounds = 5000,
                 early_stopping_rounds = 12,
                 param_sample_size = 20,
                 numeric_columns = config.contin_x_cols,
                 categorical_columns = config.categ_x_cols,
                 y_column = config.y_col,
                 numeric_transformers = [trans_mod.MissingnessIndicatorTransformer(),
                                         trans_mod.ZeroVarianceTransformer(),
                                         trans_mod.InteractionTransformer(interaction_list = config.interaction_cols),
                                         trans_mod.PolynomialTransformer(feature_power_dict =  config.polynomial_col_dict),
                                         trans_mod.CustomScalerTransformer()],
                 categorical_transformers = [trans_mod.CategoricalTransformer(),
                                             trans_mod.ZeroVarianceTransformer()],
                 kfold_results = []):
        self.x = x
        self.y = y
        self.param_dict = param_dict
        self.k_folds = k_folds
        self.n_boost_rounds = n_boost_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.param_sample_size = param_sample_size
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.y_column =  y_column
        self.numeric_transformers = numeric_transformers
        self.categorical_transformers = categorical_transformers
        self.kfold_results = kfold_results
        
    @staticmethod
    def index_slice_list(lst, indices):
        """
        Slice a list by a list of indices (positions)
        Args:
            lst (list): list to subset
            indices (list): positions to use in subsetting lst
        Returns:
            list
        """
        list_slice = itemgetter(*indices)(lst)
        if len(indices) == 1:
            return [list_slice]
        else:
            return list(list_slice)
    
    
    @staticmethod
    def unnest_list(nested_list : list):
        """
        Unnest a list of lists
        Args:
            nested_list (list): nested list of lists
        """
        return list(itertools.chain.from_iterable(nested_list))
        

    @staticmethod
    def print_timestamp_message(message : str, timestamp_format = '%Y-%m-%d %H:%M:%S'):
        """
        Print formatted timestamp followed by custom message
        Args:
            message (str): string to concatenate with timestamp
            timestamp_format (str): format for datetime string. defaults to '%Y-%m-%d %H:%M:%S'
        """
        ts_string = datetime.datetime.fromtimestamp(time.time()).strftime(timestamp_format)
        print(f'{ts_string}: {message}')
        
        
    def get_hyperparameter_sample(self):
        """
        Generate all possible hyperparameter combinations
        from self.param_dict and return a random sample
        of size self.param_sample_size, returning  a list
        of dictionaries
        """
        param_combinations = list(itertools.product(*self.param_dict.values()))
        param_sample = random.sample(param_combinations, self.param_sample_size)
        param_sample_dict_list = [dict(zip(self.param_dict.keys(), list(ps))) for ps in param_sample]
        return param_sample_dict_list
    
    
    def get_kfold_indices(self):
        """
        Get list of shuffled indices split into <k_folds> groups.
        This is used in splitting <x> during k-fold cross validation.
        """
        indices = range(self.x.shape[0])
        shuffled_indices = sklearn.utils.shuffle(indices)
        fold_positions = [shuffled_indices[i::self.k_folds] for i in range(1, self.k_folds + 1)]
        return fold_positions
    
    def run_kfold_cv(self):
        """
        Run k-fold cross validation given supplied parameter set
        and transformation pipelines
        Returns:
            pandas.DataFrame
        """
        kfold_indices = self.get_kfold_indices()
        hyperparam_sample = self.get_hyperparameter_sample()
        self.kfold_results = []
        
        for it, hp in enumerate(hyperparam_sample):
            for k in range(1, self.k_folds + 1):
                # Determine Train, Test, Validation Indices
                test_k = k
                if test_k == self.k_folds:
                    valid_k = 1
                else:
                    valid_k = test_k + 1
                train_k = [x for x in range(1, self.k_folds + 1) if x not in [test_k, valid_k]]
                train_k_str = ','.join([str(tk) for tk in train_k])
                self.print_timestamp_message(f'Starting Grid {it+1} of {self.param_sample_size}, Fold {k} ----- Train: {train_k_str}, Test: {test_k}, Validation: {valid_k}')
                
                # Create DataFrames for Train, Test Validation
                train_x = self.x.iloc[self.unnest_list([kfi for i, kfi in enumerate(kfold_indices) if i + 1 in train_k])]
                test_x = self.x.iloc[self.unnest_list([kfi for i, kfi in enumerate(kfold_indices) if i + 1 == test_k])]
                valid_x = self.x.iloc[self.unnest_list([kfi for i, kfi in enumerate(kfold_indices) if i + 1 == valid_k])]
                
                train_y = self.y.iloc[self.unnest_list([kfi for i, kfi in enumerate(kfold_indices) if i + 1 in train_k])]
                test_y = self.y.iloc[self.unnest_list([kfi for i, kfi in enumerate(kfold_indices) if i + 1 == test_k])]
                valid_y = self.y.iloc[self.unnest_list([kfi for i, kfi in enumerate(kfold_indices) if i + 1 == valid_k])]
                
                # Process Features with Pipeline
                feature_pipeline = trans_mod.FeaturePipeline(train_df = train_x,
                                                             test_df = test_x,
                                                             valid_df = valid_x,
                                                             numeric_columns = self.numeric_columns,
                                                             categorical_columns = self.categorical_columns,
                                                             numeric_transformers = self.numeric_transformers,
                                                             categorical_transformers = self.categorical_transformers)

                train_x, test_x, valid_x  = feature_pipeline.process_train_test_valid_features()
                #print(f'Train X: {train_x.shape}, Test X: {test_x.shape}, Valid X: {valid_x.shape}')
                
                # Process Response with Pipeline
                response_pipeline = trans_mod.ResponsePipeline(train_df = train_y,
                                                               test_df = test_y,
                                                               valid_df = valid_y,
                                                               response_column = self.y_column)

                train_y, test_y, valid_y  = response_pipeline.process_train_test_valid_response()
                #print(f'Train Y: {train_y.shape}, Test Y: {test_y.shape}, Valid Y: {valid_y.shape}')
                
                # Train Model
                dat_train = xgb.DMatrix(train_x, label = train_y, enable_categorical = True)
                dat_valid = xgb.DMatrix(valid_x, label = valid_y, enable_categorical = True)
                watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
                
                xgb_trn = xgb.train(params = hp,
                                    dtrain = dat_train,
                                    num_boost_round = self.n_boost_rounds,
                                    evals = watchlist,
                                    early_stopping_rounds = self.early_stopping_rounds,
                                    verbose_eval = False)
                
                # Evaluate Results on Test Set
                pred = xgb_trn.predict(xgb.DMatrix(test_x, enable_categorical = True))
                it_output = hp.copy()
                it_output['k_fold'] = k
                it_output['log_loss'] = sklearn.metrics.log_loss(test_y, pred)
                it_output['accuracy'] = np.mean([int(np.round(p,0)) == test_y.iloc[i] for i, p  in enumerate(pred)])
                self.kfold_results.append(it_output)
        output_df = pd.DataFrame(self.kfold_results)
        return output_df
    
    def get_best_params(self):
        """
        Get best hyperparameters and associated results from self.kfold_results
        which is assigned values in the get_best_params() method
        Returns:
            dict, dict
        """
        assert len(self.kfold_results) > 0, 'Execute run_kfold_cv() method prior to get_best_params()'
        result_df = pd.DataFrame(self.kfold_results)
        group_cols = [c for c in result_df.columns if c not in ['log_loss', 'accuracy', 'k_fold']]
        
        # Aggregate Results Across Folds
        result_df_agg = result_df.\
        drop(['k_fold'], axis = 1).\
        groupby(group_cols, as_index = False).\
        agg({'log_loss' : 'mean', 'accuracy' : 'mean'})
        
        # Subset Best Parameters
        best_param_df = result_df_agg[result_df_agg.log_loss == min(result_df_agg.log_loss)]
        best_param_results = best_param_df[['log_loss', 'accuracy']].to_dict('records')[0]
        best_params = best_param_df.\
        drop(['log_loss', 'accuracy'], axis = 1).\
        to_dict('records')[0]
        return best_params, best_param_results



param_dict = {'objective': ['binary:logistic'],
              'booster': ['gbtree'],
              'eval_metric': ['logloss'],
              'eta' : list(np.linspace(0.005, 0.10, 5)),
              'gamma' : [0, 1, 2, 4],
              'max_depth' : [int(x) for x in np.linspace(4, 14, 5)],
              'min_child_weight' : list(range(1, 10, 2)),
              'subsample' : list(np.linspace(0.5, 1, 5)),
              'colsample_bytree' : list(np.linspace(0.3, 1, 5))}


xgb_tuner = XgboostClassificationTuner(x = df[[c for c in df.columns if c != config.y_col]],
                                       y = df[[config.y_col]],
                                       param_dict = param_dict,
                                       k_folds = 5,
                                       n_boost_rounds = 5000,
                                       early_stopping_rounds = 12,
                                       param_sample_size = 3,
                                       numeric_columns = config.contin_x_cols,
                                       categorical_columns = config.categ_x_cols,
                                       y_column = config.y_col)



xgb_kfold_results = xgb_tuner.run_kfold_cv()
best_params, best_results = xgb_tuner.get_best_params()




"""

kfold_results = {}
xgb_tuner.run_kfold_cv() assigns values to kfold_results dict
xgb_tuner.get_best_params() references kfold_results




sample hyperparameter space
create partitions
for loop  with:
    scale pos weight performance append
performance summary aggregation & output writing
best param csv write
performance plotting
"""





### TO DO
######################################################################################################

"""
# Assertion: if param sample > n param  combinations ... 
# add process_train_test_valid_response
# response pipeline for continuous response variable?
# start xgbtuner
# write unit tests
# Work on docstrings
# Create large repo of tabular classification datasets, benchmark and catalogue best parameters


"""
















