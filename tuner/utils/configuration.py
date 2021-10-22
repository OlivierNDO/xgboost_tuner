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



### Filename & Directory Configuration
######################################################################################################
parent_directory = 'D:/xgboost_tuner/'
model_save_dir = f'{parent_directory}saved_models/'
feature_pipeline_save_path = f'{model_save_dir}feature_transformation_pipeline.pkl'
response_pipeline_save_path = f'{model_save_dir}response_transformation_pipeline.pkl'
