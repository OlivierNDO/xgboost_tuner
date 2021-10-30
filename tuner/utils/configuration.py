### Load Packages
######################################################################################################
import numpy as np


### Filename & Directory Configuration
######################################################################################################
parent_directory = 'D:/xgboost_tuner/'
model_save_dir = f'{parent_directory}saved_models/'
results_dir = f'{parent_directory}tuning_results/'
feature_pipeline_save_path = f'{model_save_dir}feature_transformation_pipeline.pkl'
response_pipeline_save_path = f'{model_save_dir}response_transformation_pipeline.pkl'


### Column Configuration
######################################################################################################
y_col = 'RETAINED'
id_col = 'FAKE_POLICY_NUMBERS'
categ_x_cols = ['PDW_PACKAGE_POLICY_FLAG', 'RATE_STATE_NAME', 'TIER_CODE',
                        'NEW_RENEWAL_CODE', 'OCCUPATION']
contin_x_cols = ['IBS_CREDIT_SCORE_NUMBER', 'ADDRESS_YEARS', 'EMPLOYED_YEARS',
                 'POLICY_TERM_PREMIUM_AMOUNT', 'POLICY_TERM_NUMBER_OF_MONTHS',
                 '0_3_MAJOR', '3_5_MAJOR', '0_3_MINOR', '3_5_MINOR', 'OPERATOR_AGE', 'VEH_ISO_LIAB_SYM']

x_cols = categ_x_cols + contin_x_cols


### Feature Transformation Configuration
######################################################################################################
interaction_cols = [('IBS_CREDIT_SCORE_NUMBER', 'OPERATOR_AGE')]

polynomial_col_dict = {'OPERATOR_AGE' : 2}


### File Paths
######################################################################################################
folder_path = 'D:/kemper_exercise/raw_data/'
train_file_name = 'retention_sample.csv'


### Crossvalidation & Hyperparameter Configuration
######################################################################################################
crossval_config = {'k_folds' : 5, 'n_boost_rounds' : 5000, 'early_stopping_rounds' : 12, 'param_sample_size' : 1000}




### Dataset Dictionaries
######################################################################################################
retention_dataset_dict = {'y_col' : 'RETAINED',
                          'id_col' : 'FAKE_POLICY_NUMBERS',
                          'categ_x_cols' : ['PDW_PACKAGE_POLICY_FLAG', 'RATE_STATE_NAME', 'TIER_CODE',
                                            'NEW_RENEWAL_CODE', 'OCCUPATION'],
                          'contin_x_cols' : ['IBS_CREDIT_SCORE_NUMBER', 'ADDRESS_YEARS', 'EMPLOYED_YEARS',
                                           'POLICY_TERM_PREMIUM_AMOUNT', 'POLICY_TERM_NUMBER_OF_MONTHS',
                                           '0_3_MAJOR', '3_5_MAJOR', '0_3_MINOR', '3_5_MINOR',
                                           'OPERATOR_AGE', 'VEH_ISO_LIAB_SYM'],
                          'x_cols' : ['PDW_PACKAGE_POLICY_FLAG', 'RATE_STATE_NAME', 'TIER_CODE',
                                      'NEW_RENEWAL_CODE', 'OCCUPATION', 'IBS_CREDIT_SCORE_NUMBER',
                                      'ADDRESS_YEARS', 'EMPLOYED_YEARS', 'POLICY_TERM_PREMIUM_AMOUNT',
                                      'POLICY_TERM_NUMBER_OF_MONTHS', '0_3_MAJOR', '3_5_MAJOR',
                                      '0_3_MINOR', '3_5_MINOR', 'OPERATOR_AGE', 'VEH_ISO_LIAB_SYM'],
                          'interaction_cols' : [('IBS_CREDIT_SCORE_NUMBER', 'OPERATOR_AGE')],
                          'polynomial_col_dict' : {'OPERATOR_AGE' : 2},
                          'folder_path' : 'D:/kemper_exercise/raw_data/',
                          'train_file_name' : 'retention_sample.csv',
                          'dataset_name' : 'retention_classification',
                          'result_folder' : results_dir,
                          'hyperparam_config' : {'objective': ['binary:logistic'],
                                               'booster': ['gbtree'],
                                               'eval_metric': ['logloss'],
                                               'num_class' : [2],
                                               'eta' : list(np.linspace(0.005, 0.1, 5)),
                                               'gamma' : [0, 1, 2, 4],
                                               'max_depth' : [int(x) for x in np.linspace(4, 14, 5)],
                                               'min_child_weight' : list(range(1, 10, 3)),
                                               'subsample' : list(np.linspace(0.5, 1, 4)),
                                               'colsample_bytree' : list(np.linspace(0.3, 1, 4))}}


sep2021_playg_dataset_dict = {'y_col' : 'claim',
                              'id_col' : 'id',
                              'categ_x_cols' : [],
                              'contin_x_cols' : ['f1', 'f10', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105',
                                                 'f106', 'f107', 'f108', 'f109', 'f11', 'f110', 'f111',
                                                 'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118',
                                                 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',
                                                 'f2', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26',
                                                 'f27', 'f28', 'f29', 'f3', 'f30', 'f31', 'f32', 'f33',
                                                 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f4', 'f40',
                                                 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48',
                                                 'f49', 'f5', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55',
                                                 'f56', 'f57', 'f58', 'f59', 'f6', 'f60', 'f61', 'f62',
                                                 'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f7',
                                                 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77',
                                                 'f78', 'f79', 'f8', 'f80', 'f81', 'f82', 'f83', 'f84',
                                                 'f85', 'f86', 'f87', 'f88', 'f89', 'f9', 'f90', 'f91',
                                                 'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99'],
                              'x_cols' : ['f1', 'f10', 'f100', 'f101', 'f102', 'f103', 'f104', 'f105',
                                         'f106', 'f107', 'f108', 'f109', 'f11', 'f110', 'f111',
                                         'f112', 'f113', 'f114', 'f115', 'f116', 'f117', 'f118',
                                         'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',
                                         'f2', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26',
                                         'f27', 'f28', 'f29', 'f3', 'f30', 'f31', 'f32', 'f33',
                                         'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f4', 'f40',
                                         'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48',
                                         'f49', 'f5', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55',
                                         'f56', 'f57', 'f58', 'f59', 'f6', 'f60', 'f61', 'f62',
                                         'f63', 'f64', 'f65', 'f66', 'f67', 'f68', 'f69', 'f7',
                                         'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77',
                                         'f78', 'f79', 'f8', 'f80', 'f81', 'f82', 'f83', 'f84',
                                         'f85', 'f86', 'f87', 'f88', 'f89', 'f9', 'f90', 'f91',
                                         'f92', 'f93', 'f94', 'f95', 'f96', 'f97', 'f98', 'f99'],
                              'interaction_cols' : [],
                              'polynomial_col_dict' : {},
                              'folder_path' : 'D:/tabular_classif/sep_2021/',
                              'train_file_name' : 'train.csv',
                              'dataset_name' : 'kaggle_tabular_playground_sep2021',
                              'result_folder' : results_dir,
                              'hyperparam_config' : {'objective': ['binary:logistic'],
                                                   'booster': ['gbtree'],
                                                   'eval_metric': ['logloss'],
                                                   'num_class' : [2],
                                                   'eta' : list(np.linspace(0.005, 0.1, 5)),
                                                   'gamma' : [0, 1, 2, 4],
                                                   'max_depth' : [int(x) for x in np.linspace(4, 14, 5)],
                                                   'min_child_weight' : list(range(1, 10, 3)),
                                                   'subsample' : list(np.linspace(0.5, 1, 4)),
                                                   'colsample_bytree' : list(np.linspace(0.3, 1, 4))}}


jun2021_playg_dataset_dict = {'y_col' : 'target',
                              'id_col' : 'id',
                              'categ_x_cols' : [],
                              'contin_x_cols' : ['feature_0', 'feature_1', 'feature_10', 'feature_11', 'feature_12', 'feature_13',
                                                 'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19',
                                                 'feature_2', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24',
                                                 'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_3',
                                                 'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35',
                                                 'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_4', 'feature_40',
                                                 'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46',
                                                 'feature_47', 'feature_48', 'feature_49', 'feature_5', 'feature_50', 'feature_51',
                                                 'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57',
                                                 'feature_58', 'feature_59', 'feature_6', 'feature_60', 'feature_61', 'feature_62',
                                                 'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68',
                                                 'feature_69', 'feature_7', 'feature_70', 'feature_71', 'feature_72', 'feature_73',
                                                 'feature_74', 'feature_8', 'feature_9'],
                              'x_cols' : ['feature_0', 'feature_1', 'feature_10', 'feature_11', 'feature_12', 'feature_13',
                                          'feature_14', 'feature_15', 'feature_16', 'feature_17', 'feature_18', 'feature_19',
                                          'feature_2', 'feature_20', 'feature_21', 'feature_22', 'feature_23', 'feature_24',
                                          'feature_25', 'feature_26', 'feature_27', 'feature_28', 'feature_29', 'feature_3',
                                          'feature_30', 'feature_31', 'feature_32', 'feature_33', 'feature_34', 'feature_35',
                                          'feature_36', 'feature_37', 'feature_38', 'feature_39', 'feature_4', 'feature_40',
                                          'feature_41', 'feature_42', 'feature_43', 'feature_44', 'feature_45', 'feature_46',
                                          'feature_47', 'feature_48', 'feature_49', 'feature_5', 'feature_50', 'feature_51',
                                          'feature_52', 'feature_53', 'feature_54', 'feature_55', 'feature_56', 'feature_57',
                                          'feature_58', 'feature_59', 'feature_6', 'feature_60', 'feature_61', 'feature_62',
                                          'feature_63', 'feature_64', 'feature_65', 'feature_66', 'feature_67', 'feature_68',
                                          'feature_69', 'feature_7', 'feature_70', 'feature_71', 'feature_72', 'feature_73',
                                           'feature_74', 'feature_8', 'feature_9'],
                              'interaction_cols' : [],
                              'polynomial_col_dict' : {},
                              'folder_path' : 'D:/tabular_classif/jun_2021/',
                              'train_file_name' : 'train.csv',
                              'dataset_name' : 'kaggle_tabular_playground_jun2021',
                              'result_folder' : results_dir,
                              'hyperparam_config' : {'objective': ['multi:softprob'],
                                                   'booster': ['gbtree'],
                                                   'eval_metric': ['mlogloss'],
                                                   'num_class' : [9],
                                                   'eta' : list(np.linspace(0.005, 0.1, 5)),
                                                   'gamma' : [0, 1, 2, 4],
                                                   'max_depth' : [int(x) for x in np.linspace(4, 14, 5)],
                                                   'min_child_weight' : list(range(1, 10, 3)),
                                                   'subsample' : list(np.linspace(0.5, 1, 4)),
                                                   'colsample_bytree' : list(np.linspace(0.3, 1, 4))}}



#temp = pd.read_csv('D:/tabular_classif/jun_2021/train.csv')
#temp_cols = sorted(list(temp.columns))

#x_cols = [c for c in temp_cols if c not in ['target',  'id']]
#data_types = [str(temp[xc].dtype) for xc in x_cols]
#n_unique = [len(np.unique(temp[xc])) for xc in x_cols]

#temp_field_df = pd.DataFrame({'col' : x_cols, 'dtype' : data_types, 'n_unique' : n_unique})


#' '.join([f"'{c}'," for c in x_cols])




#len(np.unique(temp['target']))










