"""
Primary project module for xgboost classification hyperparameter tuning

Example usage:
    ### Import Modules
    ######################################################################################################
    parent_directory = 'D:/xgboost_tuner/'
    sys.path.append(parent_directory)
    import tuner.utils.transformation_module as trans_mod
    import tuner.utils.configuration as config
    import tuner.utils.helper_module as helper
    import tuner.utils.tuner as tuner
    
    ### Dataset Dictionary
    ######################################################################################################
    use_data = config.retention_dataset_dict
    
    
    ### Read Dataset
    ######################################################################################################
    df, train, test, valid = helper.load_dataset_from_config(config_dict = use_data)
    
    
    ### XGBoost Hyperparameter Tuning with Early Stopping & Pipeline Transformations
    ######################################################################################################
    # By using a pipeline transformation during  each of K folds, cross validation results most accurately measure true out of sample performance
    
    xgb_tuner = tuner.XgboostClassificationTuner(x = df[use_data.get('x_cols')],
                                                 y = df[[use_data.get('y_col')]],
                                                 param_dict = use_data.get('hyperparam_config'),
                                                 best_param_save_name = f"{use_data.get('result_folder')}{use_data.get('dataset_name')}_best_params.csv",
                                                 kfold_result_save_name = f"{use_data.get('result_folder')}{use_data.get('dataset_name')}_kfold_results.csv",
                                                 k_folds = config.crossval_config.get('k_folds'),
                                                 n_boost_rounds = config.crossval_config.get('n_boost_rounds'),
                                                 early_stopping_rounds = config.crossval_config.get('early_stopping_rounds'),
                                                 param_sample_size = config.crossval_config.get('param_sample_size'),
                                                 numeric_columns = use_data.get('contin_x_cols'),
                                                 categorical_columns = use_data.get('categ_x_cols'),
                                                 y_column = use_data.get('y_col'),
                                                 numeric_transformers = [trans_mod.MissingnessIndicatorTransformer(),
                                                                         trans_mod.ZeroVarianceTransformer(),
                                                                         trans_mod.InteractionTransformer(interaction_list = use_data.get('interaction_cols')),
                                                                         trans_mod.PolynomialTransformer(feature_power_dict =  use_data.get('polynomial_col_dict')),
                                                                         trans_mod.CustomScalerTransformer()],
                                                 categorical_transformers = [trans_mod.CategoricalTransformer(),
                                                                             trans_mod.ZeroVarianceTransformer()])
    
    xgb_kfold_results = xgb_tuner.run_kfold_cv()
    xgb_tuner.save_results()
"""


### Import Packages
######################################################################################################
import datetime
import itertools
import numpy as np
from operator import itemgetter
import pandas as pd
import sklearn
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



### Define Functions and Classes
######################################################################################################
class XgboostClassificationTuner:
    """
    Perform k fold cross validation over a random selection of
    hyperparameter space  using train, test, and validation sets.
    Each k-fold iteration assigns 1 fold to test, 1 fold to validation,
    and remaining folds to train. A custom sklearn Pipeline object
    is required to most accurately measure true out of sample performance.
    RandomizedSearchCV and similar cross validation frameworks typically
    do not have functionality to customize each split of data with
    transformation pipelines, class weight parameters, and early stopping.
    """ 
    def __init__(self,
                 x : pd.DataFrame,
                 y : pd.DataFrame,
                 param_dict :  dict,
                 best_param_save_name = None,
                 kfold_result_save_name = None,
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
                                         trans_mod.PolynomialTransformer(feature_power_dict = config.polynomial_col_dict),
                                         trans_mod.CustomScalerTransformer()],
                 categorical_transformers = [trans_mod.CategoricalTransformer(),
                                             trans_mod.ZeroVarianceTransformer()],
                 kfold_results = []):
        self.x = x
        self.y = y
        self.param_dict = param_dict
        self.best_param_save_name = best_param_save_name
        self.kfold_result_save_name = kfold_result_save_name
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
        
        
    @staticmethod
    def get_balanced_sample_weights(y_train : pd.Series):
        """
        Create array of weights to adjust for class imbalance
        Args:
            y_train: training set response variable (list, numpy array, or pandas.Series)
        """
        train_class_counts = dict((x, len([z for z in y_train if z == x])) for x in set(y_train))
        max_class = max(train_class_counts.values())
        class_weights = [max_class / x for x in train_class_counts.values()]
        class_weight_dict = dict(zip([i for i in train_class_counts.keys()], class_weights))
        sample_weights = [class_weight_dict.get(x) for x in y_train]
        return sample_weights
        
        
    def get_hyperparameter_sample(self, print_sample = True):
        """
        Generate all possible hyperparameter combinations
        from self.param_dict and return a random sample
        of size self.param_sample_size, returning  a list
        of dictionaries
        Args:
            print_sample (bool): if True, print percentage of parameter space sampled
        """
        param_combinations = list(itertools.product(*self.param_dict.values()))
        param_sample = random.sample(param_combinations, self.param_sample_size)
        param_sample_dict_list = [dict(zip(self.param_dict.keys(), list(ps))) for ps in param_sample]
        if print_sample:
            percent_sample = self.param_sample_size / len(param_combinations)
            percent_sample_label = f'{round(percent_sample * 100,3)}%'
            self.print_timestamp_message(f'Sampling {self.param_sample_size} of {len(param_combinations)} \
                                         ({percent_sample_label}) parameter combinations')
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
                
                # Process Response with Pipeline
                response_pipeline = trans_mod.ResponsePipeline(train_df = train_y,
                                                               test_df = test_y,
                                                               valid_df = valid_y,
                                                               response_column = self.y_column)

                train_y, test_y, valid_y  = response_pipeline.process_train_test_valid_response()
                
                # Train Model
                class_weight_vec = self.get_balanced_sample_weights(train_y.iloc[:, 0])
                dat_train = xgb.DMatrix(train_x, label = train_y, enable_categorical = True, weight = class_weight_vec)
                dat_valid = xgb.DMatrix(valid_x, label = valid_y, enable_categorical = True)
                watchlist = [(dat_train, 'train'), (dat_valid, 'valid')]
                
                # Remove num_class for binary classification
                if 'num_class' in list(hp.keys()):
                    if hp.get('num_class') == 2:
                        del hp['num_class']
                
                xgb_trn = xgb.train(params = hp,
                                    dtrain = dat_train,
                                    num_boost_round = self.n_boost_rounds,
                                    evals = watchlist,
                                    early_stopping_rounds = self.early_stopping_rounds,
                                    verbose_eval = False)
                
                # Evaluate Results on Test Set
                pred = xgb_trn.predict(xgb.DMatrix(test_x, enable_categorical = True))
                it_output = hp.copy()
                y_class_labels = list(np.unique(list(train_y.iloc[:, 0]) + list(valid_y.iloc[:, 0]) + list(test_y.iloc[:, 0])))
                it_output['log_loss'] = sklearn.metrics.log_loss(test_y, pred, labels = y_class_labels)
                it_output['k_fold'] = k
                if 'num_class' in list(hp.keys()):
                    class_pred = list(itertools.chain.from_iterable([[i for i, pr in enumerate(pred_arr) if pr == max(pred_arr)]  for pred_arr in pred]))
                    it_output['accuracy'] = np.mean([int(np.round(p,0)) == test_y.iloc[i] for i, p  in enumerate(class_pred)])
                else:
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
    
    
    def save_results(self):
        """
        Get best hyperparameters from self.kfold_results and save both to csv files
        Returns:
            None
        """
        assert len(self.kfold_results) > 0, 'Execute run_kfold_cv() method prior to save_results()'
        assert self.best_param_save_name is not None, "parameter 'best_param_save_name' must not be None when executing save_results()"
        assert self.kfold_result_save_name is not None, "parameter 'kfold_result_save_name' must not be None when executing save_results()"        
        best_params, best_param_results = self.get_best_params()
        best_param_df = pd.DataFrame(best_params, index=[0])
        kfold_result_df = pd.DataFrame(self.kfold_results)
        best_param_df.to_csv(self.best_param_save_name, index = False)
        kfold_result_df.to_csv(self.kfold_result_save_name, index = False)
