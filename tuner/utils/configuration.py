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
crossval_config = {'k_folds' : 5, 'n_boost_rounds' : 5000, 'early_stopping_rounds' : 12, 'param_sample_size' : 2}




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
                                               'eta' : list(np.linspace(0.005, 0.06, 5)),
                                               'gamma' : [0, 1, 2, 4],
                                               'max_depth' : [int(x) for x in np.linspace(4, 14, 5)],
                                               'min_child_weight' : list(range(1, 10, 3)),
                                               'subsample' : list(np.linspace(0.5, 1, 4)),
                                               'colsample_bytree' : list(np.linspace(0.3, 1, 4))}}













