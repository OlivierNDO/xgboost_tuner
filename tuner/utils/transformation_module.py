### Import Packages
######################################################################################################
import numpy as np
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler, PowerTransformer, OneHotEncoder
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import _VectorizerMixin
from sklearn.feature_selection._base import SelectorMixin
from sklearn.impute import SimpleImputer



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
        output_copy = pd.concat([target_copy.fillna(0).reset_index(drop=True), target_missing_ind.reset_index(drop=True)], axis = 1)
        self.feature_names = list(target_copy.columns) + missing_ind_names
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
    Transform numeric variables using median and z-scores in a pandas.DataFrame
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
    for xgboost modelling
    """
    def __init__(self,
                 response_column : str,
                 train_df : pd.DataFrame,
                 test_df = None,
                 pipeline_save_path = None):
        self.response_column = response_column
        self.train_df = train_df
        self.test_df = test_df 
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
    """
    def __init__(self,
                 numeric_columns : list,
                 categorical_columns : list,
                 train_df : pd.DataFrame,
                 test_df = None,
                 pipeline_save_path = None,
                 numeric_transformers = [MissingnessIndicatorTransformer(),
                                         ZeroVarianceTransformer(),
                                         CustomScalerTransformer()],
                 categorical_transformers = [CategoricalTransformer(),
                                             ZeroVarianceTransformer()]):
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.train_df = train_df
        self.test_df = test_df
        self.pipeline_save_path = pipeline_save_path
        self.numeric_transformers = numeric_transformers
        self.categorical_transformers = categorical_transformers
            
    def __str__(self):
        return 'FeaturePipeline'
        
    def make_numeric_transformer(self):
        steps = [(transformer.__str__(), transformer) for transformer in self.numeric_transformers]
        numeric_transformer = Pipeline(steps = steps)
        return numeric_transformer
    
    def make_categorical_transformer(self):
        steps = [(transformer.__str__(), transformer) for transformer in self.categorical_transformers]
        categorical_transformer = Pipeline(steps = steps)
        return categorical_transformer
    
    def get_pipeline(self):
        numeric_transformer = self.make_numeric_transformer()
        categorical_transformer = self.make_categorical_transformer()
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_columns),
                                                       ('cat', categorical_transformer, self.categorical_columns)],
                                         remainder = 'passthrough')
        pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
        return pipeline
    
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
        numeric_transformer = self.make_numeric_transformer()
        categorical_transformer = self.make_categorical_transformer()
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_columns),
                                                       ('cat', categorical_transformer, self.categorical_columns)],
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
        
        return train_x, test_x
    
    def save_pipeline(self):
        # Assertions
        assert len(self.get_missing_train_cols()) == 0, f"Columns missing from training set: {self.get_missing_train_cols()}"
        assert self.pipeline_save_path is not None, 'Argument pipeline_save_path must be a string, not None type'
        
        # Define Transformation Pipeline
        numeric_transformer = self.make_numeric_transformer()
        categorical_transformer = self.make_categorical_transformer()
        preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, self.numeric_columns),
                                                       ('cat', categorical_transformer, self.categorical_columns)],
                                         remainder = 'passthrough')
        pipeline = Pipeline(steps = [('preprocessor', preprocessor)])
        
        # Fit Pipeline on Training Set & Save to pkl File
        x_cols = [c for c in self.train_df.columns if c in (self.numeric_columns + self.categorical_columns)]
        pipeline.fit(self.train_df[x_cols])
        with open(self.pipeline_save_path, 'wb') as f:
            pickle.dump(pipeline, f)
            print(f'sklearn.Pipeline object saved to {self.pipeline_save_path}')

