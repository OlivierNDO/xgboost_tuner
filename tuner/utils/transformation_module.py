"""
This is a collection of custom sklearn transformation classes and utilities for combining them
into cohesive transformation pipelines

Example usage of this module:
    import tuner.utils.transformation_module as trans_mod
    
    # Read Data
    df, train, test, valid = helper.load_dataset_from_config(config_dict = use_data)
    
    # Transform Predictor Features
    feature_pipeline = trans_mod.FeaturePipeline(train_df = train,
                                                 test_df = test,
                                                 valid_df = valid,
                                                 numeric_columns = use_data.get('contin_x_cols'),
                                                 categorical_columns = use_data.get('categ_x_cols'),
                                                 numeric_transformers = [trans_mod.MissingnessIndicatorTransformer(),
                                                                         trans_mod.ZeroVarianceTransformer(),
                                                                         trans_mod.InteractionTransformer(interaction_list = use_data.get('interaction_cols')),
                                                                         trans_mod.PolynomialTransformer(feature_power_dict =  use_data.get('polynomial_col_dict')),
                                                                         trans_mod.CustomScalerTransformer()],
                                                 categorical_transformers = [trans_mod.CategoricalTransformer(),
                                                                             trans_mod.ZeroVarianceTransformer()])
    
    train_x, test_x, valid_x  = feature_pipeline.process_train_test_valid_features()
    
    # Transform Response Variable
    response_pipeline = trans_mod.ResponsePipeline(train_df = train, test_df = test,
                                                   valid_df = valid, response_column = use_data.get('y_col'))
    
    train_y, test_y, valid_y  = response_pipeline.process_train_test_valid_response()

"""



### Import Packages
######################################################################################################
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder, MinMaxScaler
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.impute import SimpleImputer
import sys


### Import Modules
######################################################################################################
parent_directory = 'D:/xgboost_tuner/'
sys.path.append(parent_directory)
import tuner.utils.configuration as config



### Define Functions & Classes
######################################################################################################
class MissingnessIndicatorTransformer(BaseEstimator, TransformerMixin):
    """
    Replace missing values in continuous fields with zero and create missingness
    indicators in new columns. e.g. columns 'A_Missing', 'B_Missing' are
    returned in addition to 'A' and 'B'
    """
    def __init__(self, feature_names = []):
        self.feature_names = feature_names
        
    def __str__(self):
        return 'MissingnessIndicatorTransformer'
        
    @staticmethod
    def binary_missingness(iterable):
        return [1 if (np.isnan(x) or np.isinf(x)) else 0 for x in iterable]
      
    def fit(self, target):
        return self
    
    def transform(self, target):
        self.feature_names = []
        target_copy = target.copy()
        target_missing_ind_list = []
        missing_ind_names = []
        for c in target_copy.columns:
            col_c_indicator = pd.DataFrame({f'{c}_Missing' : self.binary_missingness(iterable = target_copy[c])})
            missing_ind_names.append(f'{c}_Missing')
            target_missing_ind_list.append(col_c_indicator)
        target_missing_ind = pd.concat(target_missing_ind_list, axis = 1)
        # new
        target_copy.replace([np.inf, -np.inf], np.nan, inplace=True)
        output_copy = pd.concat([target_copy.fillna(0).reset_index(drop=True), target_missing_ind.reset_index(drop=True)], axis = 1)
        self.feature_names = list(target_copy.columns) + missing_ind_names
        return output_copy



class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    One-hot encode categorical columns in a way that will work when
    the test set has values not found in the training set. Columns
    returned will exclude a single reference level. e.g.
    if a pandas.DataFrame with column 'X' and values 'A', 'B', and 'C'
    is passed where frequencies of 'A', 'B', and 'C' are 500, 250, and 250,
    respectively, binary columns 'X_B' and 'X_C' will be returned.
    """
    def __init__(self, column_value_counts = {}, feature_names = []):
        self.column_value_counts = column_value_counts
        self.feature_names = feature_names
        
    def __str__(self):
        return 'CategoricalTransformer'
      
    def fit(self, target):
        for column in target.columns:
            value_count_series = target[column].value_counts()
            key_values = list(value_count_series.index)
            key_counts = list(value_count_series.values)
            self.column_value_counts[column] = dict(zip(key_values, key_counts))
        return self

    def transform(self, target):
        self.feature_names = []
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
                self.feature_names.append(f'{ohc}_{level}')
                one_hot_df_list.append(encoded_values)
        target_copy = pd.concat(one_hot_df_list, axis = 1)
        return target_copy
    
    

class ZeroVarianceTransformer(BaseEstimator, TransformerMixin):
    """
    Removes columns in pandas.DataFrame with zero variance based on the training set.
    This is necessary after missingness indicators are created for
    every column - including columns without missing values.
    """
    def __init__(self, zero_variance_cols = [], feature_names = []):
        self.zero_variance_cols = zero_variance_cols
        self.feature_names = feature_names
        
    def __str__(self):
        return 'ZeroVarianceTransformer'
        
    def fit(self, target):
        self.zero_variance_cols = [c for c in target.columns if len(np.unique(target[c])) == 1]
        return self

    def transform(self, target):
        self.feature_names = []
        target_copy = target.copy()
        keep_cols = [c for c in target_copy.columns if c not in self.zero_variance_cols]
        self.feature_names = keep_cols
        target_copy = target_copy[keep_cols]
        return target_copy
    
    

class CustomScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Scale numeric variables using median and z-scores in a pandas.DataFrame
    """
    def __init__(self, column_statistics = {}, feature_names = []):
        self.column_statistics = column_statistics
        self.feature_names = feature_names
        
    def __str__(self):
        return 'CustomScalerTransformer'
        
    def fit(self, target):
        for c in target.columns:
            self.column_statistics[c] = {'median' : np.median(target[c]), 'standard deviation' : np.std(target[c])}
        return self

    def transform(self, target):
        target_copy = target.copy()
        self.feature_names = []
        for c in target_copy.columns:
            c_median = self.column_statistics.get(c).get('median')
            c_stdev = self.column_statistics.get(c).get('standard deviation')
            target_copy[c] = [(x - c_median) / c_stdev for x in target_copy[c]]
            self.feature_names.append(c)
        return target_copy
    
    

class PolynomialTransformer(BaseEstimator, TransformerMixin):
    """
    Removes columns in pandas.DataFrame with zero variance based on the training set.
    This is necessary after missingness indicators are created for
    every column - including columns without missing values.
    
    Args:
        feature_power_dict: dictionary formatted  as {column_name : exponential power}
    
    Example Usage:
        example_df = pd.DataFrame({'x1' : [10, 20, 30], 'x2' : [3, 6, 9]})
        poly_transformer = PolynomialTransformer(feature_power_dict = {'x1' : 2, 'x2' : 3})
        poly_transformer.fit_transform(example_df)
        
           x1  x2  x1_power2  x2_power2  x2_power3
        0  10   3        100          9         27
        1  20   6        400         36        216
        2  30   9        900         81        729
    """
    def __init__(self, feature_power_dict, feature_names = []):
        self.feature_power_dict = feature_power_dict
        self.feature_names = feature_names
        
    def __str__(self):
        return 'PolynomialTransformer'

    def fit(self, target):
        return self

    def transform(self, target):
        target_copy = target.copy()
        for c in target_copy.columns:
            if c in list(self.feature_power_dict.keys()):
                c_powers = range(2, self.feature_power_dict.get(c) + 1)
                for cp in c_powers:
                    power_colname = f'{c}_power{cp}'
                    target_copy[power_colname] = [x ** cp for x in target_copy[c]]
        self.feature_names = list(target_copy.columns)
        return target_copy[self.feature_names]



class LogNormTransformer(BaseEstimator, TransformerMixin):
    """
    Scale features to 0 -> 1 range before applying log(x+1) to each numeric value.
    A new column will be created that modifies the original column name.
    e.g. 'original_colname' -> 'original_colname_lognorm'
    """
    def __init__(self, scaler = MinMaxScaler(), feature_names = []):
        self.scaler = scaler
        self.feature_names = feature_names
        
    def __str__(self):
        return 'LogNormTransformer'

    def fit(self, target):
        self.scaler.fit(target)
        return self

    def transform(self, target):
        target_copy = target.copy()
        target_copy = pd.DataFrame(self.scaler.transform(target_copy), columns = target.columns)
        for c in target_copy.columns:
            lognorm_colname = f'{c}_lognorm'
            target_copy[lognorm_colname] = [np.log(x + 1) for x in target_copy[c]]
        self.feature_names = list(target_copy.columns)
        return target_copy[self.feature_names]



class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Create interaction features via multiplication
    Args:
        interaction_list: list of tuples with column name interactions
    Example Usage:
        example_df = pd.DataFrame({'x1' : [10, 20, 30], 'x2' : [3, 6, 9], 'x3' : [5, 10, 20]})
        intx_tran = InteractionTransformer(interaction_list = [('x1', 'x2'), ('x1', 'x3')])
        intx_tran.fit_transform(example_df)
        
           x1  x2  x3  x1_X_x2  x1_X_x3
        0  10   3   5       30       50
        1  20   6  10      120      200
        2  30   9  20      270      600
        
    """
    def __init__(self, interaction_list, feature_names = []):
        self.interaction_list = interaction_list
        self.feature_names = feature_names
    def __str__(self):
        return 'InteractionTransformer'

    def fit(self, target):
        return self

    def transform(self, target):
        target_copy = target.copy()       
        for intx in self.interaction_list:
            intx_colname = f'{intx[0]}_X_{intx[1]}'
            target_copy[intx_colname] = target_copy[intx[0]] * target_copy[intx[1]]
        self.feature_names = list(target_copy.columns)
        return target_copy[self.feature_names]
    


class ResponseTransformer(BaseEstimator, TransformerMixin):
    """
    Transform pandas.Series response variable to a set of integers
    (if needed) for xgboost modelling
    """
    def __init__(self, class_dictionary = {}):
        self.class_dictionary = class_dictionary
        
    def __str__(self):
        return 'ResponseTransformer'
        
    def fit(self, target):
        if pd.api.types.is_string_dtype(target):
            unique_values = list(np.unique(target))
            integer_elements = list(range(len(unique_values)))
            self.class_dictionary = dict(zip(unique_values, integer_elements))
        else:
            unique_values = list(np.unique(target))
            self.class_dictionary = dict(zip(unique_values, unique_values))
        return self
    
    def transform(self, target):
        return_target = pd.DataFrame({target.name : [self.class_dictionary.get(x) for x in target]})
        return return_target
    
    

class ResponsePipeline:
    """
    Wrapper around ResponseTransformer class intended to transform
    pandas.Series response variable to a set of integers (if needed)
    for xgboost modeling
    """
    def __init__(self,
                 response_column : str,
                 train_df : pd.DataFrame,
                 test_df = None,
                 valid_df = None,
                 pipeline_save_path = None):
        self.response_column = response_column
        self.train_df = train_df
        self.test_df = test_df
        self.valid_df = valid_df
        self.pipeline_save_path = pipeline_save_path
        
        
    def __str__(self):
        return 'ResponsePipeline'
        
    def process_train_test_response(self):
        # Assertions
        assert self.test_df is not None, 'Error: Parameter test_df cannot be None when calling method process_train_test_features()'
        assert self.response_column in self.train_df.columns, f"Column {self.respones_column} missing from training set"
        assert self.response_column in self.test_df.columns, f"Column {self.respones_column} missing from test set"
        
        # Define & Apply Transformation Pipeline
        response_transformer = ResponseTransformer()
        train_y = response_transformer.fit_transform(self.train_df[self.response_column])
        test_y = response_transformer.transform(self.test_df[self.response_column])
        return train_y, test_y
    
    def process_train_test_valid_response(self):
        # Assertions
        assert self.test_df is not None, 'Error: Parameter test_df cannot be None when calling method process_train_test_response()'
        assert self.valid_df is not None, 'Error: Parameter valid_df cannot be None when calling method process_train_test_valid_response()'
        assert self.response_column in self.train_df.columns, f"Column {self.respones_column} missing from training set"
        assert self.response_column in self.test_df.columns, f"Column {self.respones_column} missing from test set"
        
        # Define & Apply Transformation Pipeline
        response_transformer = ResponseTransformer()
        train_y = response_transformer.fit_transform(self.train_df[self.response_column])
        test_y = response_transformer.transform(self.test_df[self.response_column])
        valid_y = response_transformer.transform(self.valid_df[self.response_column])
        return train_y, test_y, valid_y
    
    def save_pipeline(self):
        response_transformer = ResponseTransformer()
        response_transformer.fit(self.train_df[self.response_column])
        with open(self.pipeline_save_path, 'wb') as f:
            pickle.dump(response_transformer, f)
            print(f'sklearn.Pipeline object saved to {self.pipeline_save_path}')
    


class FeaturePipeline:
    """
    Transform predictor features in a pandas.DataFrame using sklearn.ColumnTransformer
    and sklearn.Pipeline objects. This class serves as a wrapper around
    the following classes:
        > MissingnessIndicatorTransformer()
        > CategoricalTransformer()
        > ZeroVarianceTransformer()
        > CustomScalerTransformer()
        > InteractionTransformer()
        > ZeroVarianceTransformer()
        
    Note that any custom sklearn transformer class objects can be passed
    to numeric_transformers and categorical_transformers objects
    """
    def __init__(self,
                 numeric_columns : list,
                 categorical_columns : list,
                 train_df : pd.DataFrame,
                 test_df = None,
                 valid_df = None,
                 pipeline_save_path = None,
                 numeric_transformers = [MissingnessIndicatorTransformer(),
                                         ZeroVarianceTransformer(),
                                         InteractionTransformer(interaction_list = config.interaction_cols),
                                         PolynomialTransformer(feature_power_dict =  config.polynomial_col_dict),
                                         CustomScalerTransformer()],
                 categorical_transformers = [CategoricalTransformer(),
                                             ZeroVarianceTransformer()]):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.train_df = train_df
        self.test_df = test_df
        self.valid_df = valid_df
        self.pipeline_save_path = pipeline_save_path
        self.numeric_transformers = numeric_transformers
        self.categorical_transformers = categorical_transformers
            
    def __str__(self):
        return 'FeaturePipeline'
        
    def make_transformer_pipeline(self, transformer_list):
        steps = [(transformer.__str__(), transformer) for transformer in transformer_list]
        transformer_pipeline = Pipeline(steps = steps)
        return transformer_pipeline
    
    def get_missing_train_cols(self):
        x_cols = self.numeric_columns + self.categorical_columns
        missing_cols = [xc for xc in x_cols if xc not in self.train_df.columns]
        return missing_cols
    
    def get_missing_test_cols(self):
        x_cols = self.numeric_columns + self.categorical_columns
        missing_cols = [xc for xc in x_cols if xc not in self.test_df.columns]
        return missing_cols
        
    def process_train_test_features(self):
        # Assertions
        assert self.test_df is not None, 'Error: Parameter test_df cannot be None when calling method process_train_test_features()'
        assert len(self.get_missing_train_cols()) == 0, f"Columns missing from training set: {self.get_missing_train_cols()}"
        assert len(self.get_missing_test_cols()) == 0, f"Columns missing from test set: {self.get_missing_test_cols()}"
        
        # Define Transformation Pipeline
        preprocessor = ColumnTransformer(transformers=[('num', self.make_transformer_pipeline(transformer_list = self.numeric_transformers), self.numeric_columns),
                                                       ('cat', self.make_transformer_pipeline(transformer_list = self.categorical_transformers), self.categorical_columns)],
                                         remainder = 'passthrough')
        pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
        
        
        # Apply Transformation to Train & Test
        x_cols = [c for c in self.train_df.columns if c in (self.numeric_columns + self.categorical_columns)]
        train_x = pipeline.fit_transform(self.train_df[x_cols])
        test_x = pipeline.transform(self.test_df[x_cols])
        
        # Retrieve Feature Names from Preprocessor
        feature_name_list = []
        for i, transf in enumerate(preprocessor.transformers_):
            last_transf_step = transf[1].steps[-1][1]
            feature_name_list = feature_name_list + last_transf_step.feature_names
        train_x = pd.DataFrame(train_x, columns = feature_name_list)
        test_x = pd.DataFrame(test_x, columns = feature_name_list)
        
        # Remove Duplicate Columns
        train_x = train_x.loc[:,~train_x.columns.duplicated()]
        test_x = test_x.loc[:,~test_x.columns.duplicated()]
        return train_x, test_x
    
    def process_train_test_valid_features(self):
        # Assertions
        assert self.test_df is not None, 'Error: Parameter test_df cannot be None when calling method process_train_test_features()'
        assert self.valid_df is not None, 'Error: Parameter valid_df cannot be None when calling method process_train_test_features()'
        assert len(self.get_missing_train_cols()) == 0, f"Columns missing from training set: {self.get_missing_train_cols()}"
        assert len(self.get_missing_test_cols()) == 0, f"Columns missing from test set: {self.get_missing_test_cols()}"
        
        # Define Transformation Pipeline
        preprocessor = ColumnTransformer(transformers=[('num', self.make_transformer_pipeline(transformer_list = self.numeric_transformers), self.numeric_columns),
                                                       ('cat', self.make_transformer_pipeline(transformer_list = self.categorical_transformers), self.categorical_columns)],
                                         remainder = 'passthrough')
        pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
        
        
        # Apply Transformation to Train & Test
        x_cols = [c for c in self.train_df.columns if c in (self.numeric_columns + self.categorical_columns)]
        train_x = pipeline.fit_transform(self.train_df[x_cols])
        test_x = pipeline.transform(self.test_df[x_cols])
        valid_x = pipeline.transform(self.valid_df[x_cols])
        
        # Retrieve Feature Names from Preprocessor
        feature_name_list = []
        for i, transf in enumerate(preprocessor.transformers_):
            last_transf_step = transf[1].steps[-1][1]
            feature_name_list = feature_name_list + last_transf_step.feature_names
        train_x = pd.DataFrame(train_x, columns = feature_name_list)
        test_x = pd.DataFrame(test_x, columns = feature_name_list)
        valid_x = pd.DataFrame(valid_x, columns = feature_name_list)
        
        # Remove Duplicate Columns
        train_x = train_x.loc[:,~train_x.columns.duplicated()]
        test_x = test_x.loc[:,~test_x.columns.duplicated()]
        valid_x = valid_x.loc[:,~valid_x.columns.duplicated()]
        return train_x, test_x, valid_x
    
    def save_pipeline(self):
        # Assertions
        assert len(self.get_missing_train_cols()) == 0, f"Columns missing from training set: {self.get_missing_train_cols()}"
        assert self.pipeline_save_path is not None, 'Argument pipeline_save_path must be a string, not None type'
        
        # Define Transformation Pipeline
        preprocessor = ColumnTransformer(transformers=[('num', self.make_transformer_pipeline(transformer_list = self.numeric_transformers), self.numeric_columns),
                                                       ('cat', self.make_transformer_pipeline(transformer_list = self.categorical_transformers), self.categorical_columns)],
                                         remainder = 'passthrough')
        pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
        
        # Fit Pipeline on Training Set & Save to pkl File
        x_cols = [c for c in self.train_df.columns if c in (self.numeric_columns + self.categorical_columns)]
        pipeline.fit(self.train_df[x_cols])
        with open(self.pipeline_save_path, 'wb') as f:
            pickle.dump(pipeline, f)
            print(f'sklearn.Pipeline object saved to {self.pipeline_save_path}')

































