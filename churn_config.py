"""
Configuration Parameters for Chrun modules


Author: K. Bardool
Date  : Jan 2022
"""

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

RESPONSE_COL = 'Churn'

PARAM_GRID = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
}

EDA_IMAGES_PATH = './images/eda/'
RESULTS_PATH    = './images/results/'
MODELS_PATH     = './models/'
LOGGING_PATH    = './logs/'
INPUT_PATH      = './data/'

INPUT_FILE      = 'bank_data.csv'
LOGGING_FILE    = 'churn_library.log'

RF_MDL_FILE     = 'rf_model.pkl'
RF_CLS_RPT      = 'random_forest_classification_report.png'
RF_FI_RPT       = 'random_forest_feature_importance.png'

LR_MDL_FILE     = 'lr_model.pkl'
LR_CLS_RPT      = 'logistic_regr_classification_report.png'

ROC_PLOTS_RPT   = 'roc_plots.png'

TEST_SIZE = 0.3
RANDOM_SEED = 42
