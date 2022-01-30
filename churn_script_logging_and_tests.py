"""
churn_script_logging_and_testing

script to run tests against churn_library modules, and logging the results to a log file

Author: K. Bardool
Date  : Jan 2022
"""
import os
import sys
import logging
from datetime import datetime
import sklearn
import churn_library as cls
import churn_config as cfg

timestring = lambda :datetime.now().strftime('%F %H:%M:%S:%f')

logging.basicConfig(
    filename=os.path.join(cfg.LOGGING_PATH,cfg.LOGGING_FILE),
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(funcName)s - line:  %(lineno)d- %(levelname)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')


def test_import(import_data):
    '''
    test data_import  module by calling it under various test conditions:

    Test 1: call with no input parms                                    - Should fail
    Test 2: call passing a valid dataset name as input parm column name - Should succeed
    test 3: verify returned dataframe contains rows/columns             - Should succeed
    '''
    logging.info("Testing import_data()")

    ## Test 1
    logging.info("Testing import_data(): TEST 1 ")
    try:
        dframe = import_data()
        logging.info("Testing import_data(): TEST 1 - Passed -  import_data"
                     " failed to raise error when called with no input parm")
    except AssertionError:
        logging.info("Testing import_data(): TEST 1 - Failed - import_data"
                     " raised error called with no input parm")

    ## Test 2
    logging.info("Testing import_data(): TEST 2 ")
    input_file = os.path.join(cfg.INPUT_PATH, cfg.INPUT_FILE)
    try:
        dframe = import_data(input_file)
        logging.info("Testing import_data(): TEST 2 - Passed")
    except FileNotFoundError as err:
        logging.error("Testing import_data(): TEST 2 - Failed - "
                      "The provided input file wasn't found")
        raise err

    ## Test 3
    logging.info("Testing import_data(): TEST 3 ")
    try:
        assert dframe.shape[0] > 0
        assert dframe.shape[1] > 0
        logging.info(
            "Testing import_data(): TEST 3 - Passed - input file has %s rows and %s columns ",
            dframe.shape[0],
            dframe.shape[0])
    except AssertionError:
        logging.error(
            "Testing import_data(): TEST 3 - Failed - "
            "The file doesn't appear to have rows and columns")


def test_eda(perform_eda):
    '''
    test perform_eda  module by calling it under various test conditions:

    Test 1: call with valid dataframe                                           - Should succeed
    Test 2-7: verify all eda plots have been successfully written to images/eda - Should succeed
    '''

    logging.info("Testing perform_eda()")
    input_file = os.path.join(cfg.INPUT_PATH, cfg.INPUT_FILE)
    dframe = cls.import_data(input_file)

    output_files = ['churn_distribution.png',
                    'customer_age_distribution.png',
                    'marital_status_distribution.png',
                    'total_transaction_distribution.png',
                    'correlation_heatmap.png',
                    'card_category_distribution.png']

    ## Test 1
    logging.info("Testing perform_eda(): TEST 1 ")
    try:
        perform_eda(dframe)
        logging.info("Testing perform_eda(): TEST 1 - Passed - "
                     "perform_eda completed successfully")
    except Exception as err:
        logging.error("Testing perform_eda(): TEST 1 - Failed - "
                      "Exception encountered during call to perform_eda()")
        for msg in sys.exc_info():
            logging.error("Testing perform_eda(): %s", msg)
        raise Exception from err

    ## Test 2-n
    for idx, file_name in enumerate(output_files, 2):
        logging.info("Testing perform_eda(): TEST %d ", idx)
        try:
            assert os.path.exists(os.path.join(cfg.EDA_IMAGES_PATH, file_name))
            logging.info(
                "Testing perform_eda(): TEST %d - Passed - %s exists",
                idx,
                file_name)
        except AssertionError:
            logging.error(
                "Testing perform_eda(): TEST %d - Failed - %s does not exist",
                idx,
                file_name)


def test_encoder_helper(encoder_helper):
    '''
    test encoder_helper module by calling it under various test conditions:

    Test 1: call without required parameters                                      - Should fail
    Test 2: call passing an invalid column name as category_list parm             - Should fail
    Test 3: call passing an invalid column name as response parm                  - Should fail
    Test 4: call using valid parameters for both category_list and response parms - Should succeed
    '''

    logging.info("Testing encoder_helper()")
    input_file = os.path.join(cfg.INPUT_PATH, cfg.INPUT_FILE)
    dframe = cls.import_data(input_file)

    ## Test 1
    logging.info("Testing encoder_helper(): TEST 1 ")
    try:
        encoder_helper(dframe)
        logging.info("Testing encoder_helper(): TEST 1 - Passed")
    except TypeError as err:
        logging.error("Testing encoder_helper(): TEST 1 - Failed - %s", err.args)
    except Exception as err: # pylint: disable=broad-except
        logging.error(
            "Testing encoder_helper():  TEST 1 - Failed - %s",
            err.args[0])
        for msg in sys.exc_info():
            logging.error(" %s ", msg)

    ## Test 2
    logging.info("Testing encoder_helper(): TEST 2 ")
    try:
        encoder_helper(dframe, category_list = ['some_nonexisting_column'])
        logging.info(
            "Testing encoder_helper(): TEST 2 - Passed")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper(): TEST 2 - Failed - Assertion Error - "
            "invalid column name in category_list param")

    ## Test 3
    logging.info("Testing encoder_helper(): TEST 3 ")
    bad_response_colname = 'some_nonexisting_column'
    try:
        encoder_helper(dframe, category_list=cfg.CAT_COLUMNS, response=bad_response_colname)
        logging.info(
            "Testing encoder_helper(): TEST 3 - Passed")
    except AssertionError:
        logging.error(
            "Testing encoder_helper(): TEST 3 - Failed - Assertion Error - "
            "response column '%s' doesnt exist on dataframe", bad_response_colname)


    ## Test 4
    logging.info("Testing encoder_helper(): TEST 4 ")
    good_colname = 'Churn'
    try:
        encoder_helper(dframe, category_list=cfg.CAT_COLUMNS, response=good_colname)
        logging.info(
            "Testing encoder_helper(): TEST 4 - Passed")
    except AssertionError:
        logging.error(
            "Testing encoder_helper(): TEST 4 - Failed - Assertion Error - "
            "response column <%s> doesnt exist on dataframe", good_colname)


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering module by calling it under various test conditions:

    Test 1: call passing only valid dataframe                    - Should succeed
    Test 2: call passing a  invalid column name as response parm - Should fail
    Test 3: call passing a  valid column name as response parm   - Should succeed
    Test 4: verify X_train, y_train shapes match                 - Should succeed
    Test 5: verify X_test , y_test shapes match                  - Should succeed
    '''


    logging.info("Testing perform_feature_engineering()")
    input_file = os.path.join(cfg.INPUT_PATH, cfg.INPUT_FILE)
    dframe = cls.import_data(input_file)
    dframe = cls.encoder_helper(dframe, cfg.CAT_COLUMNS)


    ## Test 1 -  Call with minimal params
    logging.info("Testing perform_feature_engineering(): TEST 1")
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(dframe)
        logging.info(
            "Testing perform_feature_engineering(): TEST 1 - Passed - "
            "perform_feature_engineering successfully called")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering(): TEST 1 - Failed - Assertion Error ")

    ## Test 2 -  Call with invalid col name
    logging.info("Testing perform_feature_engineering(): TEST 2")
    bad_col_name = 'SomeName'
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dframe, response=bad_col_name)
        logging.info(
            "Testing perform_feature_engineering(): TEST 2 - Passed - "
            "perform_feature_engineering successfully called")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering(): TEST 2 - Failed - Assertion Error - "
            "response column <%s> not found in dataframe", bad_col_name)

    ## Test 3 -  Call with valid col name
    logging.info("Testing perform_feature_engineering(): TEST 3 ")
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            dframe, response=cfg.RESPONSE_COL)
        logging.info(
            "Testing perform_feature_engineering(): TEST 3 - Passed - "
            "perform_feature_engineering successfully called")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering(): TEST 3 - Failed - Assertion Error - "
            "response column <%s> not found in dataframe", bad_col_name)

    ## Test 4 -  Verify X_train , y_train shapes match
    logging.info("Testing perform_feature_engineering(): TEST 4 ")
    try:
        assert x_train.shape[0] == y_train.shape[0]
        logging.info(
            "Testing perform_feature_engineering(): TEST 4 - Passed - "
            "X_train and y_train shape match {X_train.shape[0]} = {y_train.shape[0]}")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering(): TEST 4 - Failed - "
            "Assertion Error - X_train/y_train shape mismatch %d != %d",
            x_train.shape[0],
            y_train.shape[0])

    ## Test 5 -  Verify X_test , y_test shapes match
    logging.info("Testing perform_feature_engineering(): TEST 5 ")
    try:
        assert x_test.shape[0] == y_test.shape[0]
        logging.info(
            "Testing perform_feature_engineering(): TEST 5 - Passed -"
            " X_test and y_test shape match %d = %d ",
            x_test.shape[0] , y_test.shape[0])
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering(): TEST 5 - Failed - "
            "Assertion Error - X_test and y_test shape mismatch %d != %d ",
            x_test.shape[0], y_test.shape[0])


def test_train_models(train_models):
    '''
    test train_models by calling it under various test conditions:

    Test 1: call passing valid params - Should Succeed
    Test 2: verify returned rf_model is a Random Forest classifier - Should Succeed
    Test 3: verify returned lr_model is a Logistic Regression      - Should Succeed
    Test 4: verify rf_model was successfully written (./models/rf_model.pkl) - Should Succeed
    Test 5: verify lr_model was successfully written (./models/lr_model.pkl) - Should Succeed
    '''

    logging.info("Testing train_models()")
    input_file = os.path.join(cfg.INPUT_PATH, cfg.INPUT_FILE)
    dframe = cls.import_data(input_file)
    dframe = cls.encoder_helper(dframe, cfg.CAT_COLUMNS)

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        dframe, response=cfg.RESPONSE_COL)


    ## Test 1
    logging.info("Testing train_models(): TEST 1 ")
    try:
        rf_model, lr_model = train_models(x_train, x_test, y_train, y_test)
        logging.info(
            "Testing train_models(): TEST 1 - Passed - Call sucessful")
    except Exception as err:  # pylint: disable=broad-except
        logging.error(
            "Testing train_models(): TEST 1 - Failed -  %s", err.args[0])
        for msg in sys.exc_info():
            logging.error(" %s " , msg)

    ## Test 2
    logging.info("Testing train_models(): TEST 2 ")
    try:
        assert isinstance(
            rf_model,
            sklearn.ensemble._forest.RandomForestClassifier)
        logging.info(
            "Testing train_models(): TEST 2 - Passed - Assertion Error - "
            "rf_model type is expected RandomForest ")
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 2 - Failed - Assertion Error - "
            "rf_model type mismatch: expected RandomForest classifier, recevied {type(rf_model)}")

    ## Test 3
    logging.info("Testing train_models(): TEST 3 ")
    try:
        assert isinstance(
            lr_model,
            sklearn.linear_model._logistic.LogisticRegression)
        logging.info(
            "Testing train_models(): TEST 3 - Passed - Assertion Error - "
            "lr_model type is expected LogisticRegression ")
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 3 - Failed - Assertion Error -"
            " rf_model type: expected LogisticRegression classifier, recevied {type(rf_model)}")

    ## Test 4
    logging.info("Testing train_models(): TEST 4 ")
    file_name = cfg.RF_MDL_FILE
    try:
        assert os.path.exists(os.path.join(cfg.MODELS_PATH, file_name))
        logging.info(
            "Testing train_models(): TEST 4 - Passed - %s exists", file_name)
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 4 - Failed - %s does not exist", file_name)

    ## Test 5
    logging.info("Testing train_models(): TEST 5 ")
    file_name = cfg.LR_MDL_FILE
    try:
        assert os.path.exists(os.path.join(cfg.MODELS_PATH, file_name))
        logging.info(
            "Testing train_models(): TEST 5 - Passed - %s exists", file_name)
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 5 - Failed - %s does not exist", file_name)

    ## Test 6
    logging.info("Testing train_models(): TEST 6 ")
    file_name = cfg.RF_CLS_RPT
    try:
        assert os.path.exists(os.path.join(cfg.RESULTS_PATH, file_name))
        logging.info(
            "Testing train_models(): TEST 6 - Passed - %s exists", file_name)
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 6 - Failed - %s does not exist", file_name)

    ## Test 7
    logging.info("Testing train_models(): TEST 7 ")
    file_name = cfg.LR_CLS_RPT
    try:
        assert os.path.exists(os.path.join(cfg.RESULTS_PATH, file_name))
        logging.info(
            "Testing train_models(): TEST 7 - Passed - %s exists", file_name)
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 7 - Failed - %s does not exist", file_name)

    ## Test 8
    logging.info("Testing train_models(): TEST 8 ")
    file_name = cfg.RF_FI_RPT
    try:
        assert os.path.exists(os.path.join(cfg.RESULTS_PATH, file_name))
        logging.info(
            "Testing train_models(): TEST 8 - Passed - %s exists", file_name)
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 8 - Failed - %s does not exist", file_name)

    ## Test 9
    logging.info("Testing train_models(): TEST 9 ")
    file_name = cfg.ROC_PLOTS_RPT
    try:
        assert os.path.exists(os.path.join(cfg.RESULTS_PATH, file_name))
        logging.info(
            "Testing train_models(): TEST 9 - Passed - %s exists", file_name)
    except AssertionError:
        logging.error(
            "Testing train_models(): TEST 9 - Failed - %s does not exist", file_name)


if __name__ == "__main__":
    print(f" {timestring()} - Churn library moudles test start..")
    test_import(cls.import_data)
    print(f" {timestring()} - Test import_data() completed.... ")

    test_eda(cls.perform_eda)
    print(f" {timestring()} - Test perform_eda() completed.... ")

    test_encoder_helper(cls.encoder_helper)
    print(f" {timestring()} - Test encoder_helper() completed.... ")

    test_perform_feature_engineering(cls.perform_feature_engineering)
    print(f" {timestring()} - Test perform_feature_engineering() completed.... ")

    print(f" {timestring()} - Test train_models() start.... ")
    test_train_models(cls.train_models)
    print(f" {timestring()} - Test train_models() completed.... ")

    print(f" {timestring()} - Churn library modules test complete..")
