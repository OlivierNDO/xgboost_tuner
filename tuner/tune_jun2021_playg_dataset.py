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
import tuner.utils.helper_module as helper
import tuner.utils.tuner as tuner

### Dataset Dictionary
######################################################################################################
use_data = config.jun2021_playg_dataset_dict


### Read Dataset
######################################################################################################
df, train, test, valid = helper.load_dataset_from_config(config_dict = use_data)



### XGBoost Hyperparameter Tuning with Early Stopping & Pipeline Transformations
######################################################################################################
# By using a pipeline transformation during  each of K folds, cross validation results most accurately measure true out of sample performance

xgb_tuner = tuner.XgboostClassificationTuner(x = df[use_data.get('x_cols')],
                                             y = df[[use_data.get('y_col')]],
                                             param_dict = use_data.get('hyperparam_config'),
                                             best_param_save_name = f"{use_data.get('result_folder')}{use_data.get('dataset_name')}_best_params_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                             kfold_result_save_name = f"{use_data.get('result_folder')}{use_data.get('dataset_name')}_kfold_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                             k_folds = config.crossval_config.get('k_folds'),
                                             n_boost_rounds = config.crossval_config.get('n_boost_rounds'),
                                             early_stopping_rounds = config.crossval_config.get('early_stopping_rounds'),
                                             param_sample_size = 80,
                                             numeric_columns = use_data.get('contin_x_cols'),
                                             categorical_columns = use_data.get('categ_x_cols'),
                                             y_column = use_data.get('y_col'),
                                             numeric_transformers = [trans_mod.MissingnessIndicatorTransformer(),
                                                                     trans_mod.ZeroVarianceTransformer(),
                                                                     trans_mod.CustomScalerTransformer()],
                                             categorical_transformers = [trans_mod.CategoricalTransformer(),
                                                                         trans_mod.ZeroVarianceTransformer()])


#xgb_tuner.param_dict.get('num_class')[0]


xgb_kfold_results = xgb_tuner.run_kfold_cv()
xgb_tuner.save_results()


# TO DO: update xgboost to version with gpu support
# TO DO: create unit test input files for multiclass and binary predictions

"""
Traceback (most recent call last):

  File "<ipython-input-2-a52247a02ab9>", line 19, in <module>
    xgb_kfold_results = xgb_tuner.run_kfold_cv()

  File "D:/xgboost_tuner\tuner\utils\tuner.py", line 290, in run_kfold_cv
    it_output['accuracy'] = np.mean([int(np.round(p,0)) == test_y.iloc[i] for i, p  in enumerate(class_pred)])

  File "D:/xgboost_tuner\tuner\utils\tuner.py", line 290, in <listcomp>
    it_output['accuracy'] = np.mean([int(np.round(p,0)) == test_y.iloc[i] for i, p  in enumerate(class_pred)])

  File "C:\Users\user\AppData\Roaming\Python\Python37\site-packages\pandas\core\indexing.py", line 895, in __getitem__
    return self._getitem_axis(maybe_callable, axis=axis)

  File "C:\Users\user\AppData\Roaming\Python\Python37\site-packages\pandas\core\indexing.py", line 1501, in _getitem_axis
    self._validate_integer(key, axis)

  File "C:\Users\user\AppData\Roaming\Python\Python37\site-packages\pandas\core\indexing.py", line 1444, in _validate_integer
    raise IndexError("single positional indexer is out-of-bounds")

IndexError: single positional indexer is out-of-bounds
"""
