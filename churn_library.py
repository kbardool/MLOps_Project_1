"""
churn library

This Library contains the code provided in the churn notebook which has been
refactored into a production-ready format, following PEP8 formatting guidelines.

Author: Kevin Bardool
Date  : January 2022

"""


# import libraries
import os
import sys
import logging
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import churn_config as cfg


def import_data(pth=None):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df  : pandas dataframe if csv read succeeds
                  None if csv read fails
    '''
    dframe = None
    try:
        assert pth is not None

    except AssertionError as excp:
        logging.error(
            " Assertion Error - Missing input parameter: path - path to input CSV file")
        raise AssertionError from excp

    try:
        dframe = pd.read_csv(pth)
        dframe['Churn'] = dframe['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
    except FileNotFoundError as excp:
        logging.error(" File not found %s", pth)
        logging.error(" %s ", excp.args[0])
        raise FileNotFoundError from excp
    else:
        return dframe


def perform_eda(dframe):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:  writes following diagrams to ./images/eda
            Churn Distribution
            Marital Status Distribution
            Customer Age Distribution
            Total Transaction Distribution
            Correlation Heatmap


    '''
    eda_path = cfg.EDA_IMAGES_PATH

    fig = plt.figure(figsize=(10, 5))
    dframe['Churn'].plot.hist(fig=fig, title="Churn Distribution")
    try:
        fig.savefig(os.path.join(eda_path, 'churn_distribution.png'))
    except Exception as excp:
        logging.error(" in saving chrun_distribution.png - %s ", excp.args[0])
        raise excp

    fig = plt.figure(figsize=(10, 5))
    dframe['Customer_Age'].plot.hist(fig = fig, title="Customer Age Distribution")
    try:
        fig.savefig(os.path.join(eda_path, 'customer_age_distribution.png'))
    except Exception as excp:
        logging.error(
            "  in saving customer_age_distribution.png - %s ", excp.args[0])
        raise excp

    fig = plt.figure(figsize=(10, 5))
    dframe.Marital_Status.value_counts('normalize').plot(
        kind='bar', fig=fig, title="Marital Status Distribution")
    try:
        fig.savefig(os.path.join(eda_path, 'marital_status_distribution.png'))
    except Exception as excp:
        logging.error(
            " in saving marital_status_distribution.png - %s", excp.args[0])
        raise excp

    fig = plt.figure(figsize=(10, 5))
    dframe.Card_Category.value_counts().plot(
        kind='bar', fig=fig, title="Card Category Distribution")
    try:
        fig.savefig(os.path.join(eda_path, 'card_category_distribution.png'))
    except Exception as excp:
        logging.error(
            " in saving card_category_distribution.png - %s", excp.args[0])
        raise excp

    fig = plt.figure(figsize=(10, 5))
    sns.histplot(dframe['Total_Trans_Ct'], kde=True).set_title(
        'Total Transaction Distribution')
    try:
        plt.savefig(
            os.path.join(
                eda_path,
                'total_transaction_distribution.png'))
    except Exception as excp:
        logging.error(
            " in saving total_transaction_distribution.png - %s", excp.args[0])
        raise excp

    fig = plt.figure(figsize=(20, 10))
    sns.heatmap(dframe.corr(), annot=False, cmap='Dark2_r',
                linewidths=2).set_title("Correlation Heatmap")
    try:
        plt.savefig(os.path.join(eda_path, 'correlation_heatmap.png'))
    except Exception as excp:
        logging.error(" in saving correlation_heatmap.png - %s", excp.args[0])
        raise excp

    return 0


def encoder_helper(dframe, category_list=None, response=None):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            response: string of response name [optional argument that could be used for
                      naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    if response is None:
        response = cfg.RESPONSE_COL

    if category_list is None:
        category_list = cfg.CAT_COLUMNS

    for col in category_list:
        try:
            assert col in dframe
        except AssertionError as excp:
            logging.error(
                " Assertion Error - category_list item '%s' not found in dataframe", col)
            raise AssertionError from excp

    try:
        assert response in dframe
    except AssertionError as excp:
        logging.error(
            " Assertion Error - response column '%s'not found in dataframe", response)
        raise AssertionError from excp


    quant_col_list = [f"{col}_{response}" for col in category_list]


    # Loop through columns in category_lst, and convert to numerical val
    for col, new_col_name in zip(category_list, quant_col_list):
        tmp_list = []
        try:
            categorical_groups = dframe.groupby(col).mean()[response]

            for val in dframe[col]:
                tmp_list.append(categorical_groups.loc[val])

            dframe[new_col_name] = tmp_list
        except KeyError as excp:
            logging.error(" column name: '%s'  not found in dataframe", col)
            raise excp
        except Exception as excp:
            logging.error(" ERROR: Exception encountered - %s ", excp.args[0])
            raise excp

    return dframe


def perform_feature_engineering(dframe, keep_columns = None, response= None):
    '''
    extract quantitative features into model dataset, and split dataset based on
    cfg.TEST_SIZE parameter.

    input:
              df: pandas dataframe
              response: string of response name [optional argument
              that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test:  X testing data
              y_train: y training data
              y_test:  y testing data
    '''
    if keep_columns is None:
        keep_columns = cfg.KEEP_COLS

    if response is None:
        response = cfg.RESPONSE_COL

    try:
        assert response in dframe
    except AssertionError as excp:
        logging.error(" response parm column %s not found in dataframe ", response)
        raise AssertionError from excp

    x_data = pd.DataFrame()
    y_data = dframe[response]

    for col in keep_columns:
        try:
            x_data[col] = dframe[col]
        except KeyError as excp:
            logging.error(
                " perform_feature_engineering() - column %s not found in dataframe ",col)
            raise KeyError from excp

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=cfg.TEST_SIZE, random_state=cfg.RANDOM_SEED)

    return x_train, x_test, y_train, y_test


def train_models(x_train, x_test, y_train, y_test):
    '''
    train Random Forest and Logistical Regression models
    save  generated models
    create model performance reports

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              rf_model : Trained Random Forest best estimator model
              lr_model : Trained Logistic Regression model

    as an expansion, additional models can be added
    '''

    # Random Forest Classifier
    rfc = RandomForestClassifier(random_state=cfg.RANDOM_SEED)
    # grid search
    print(" Random Forest Parameter Grid Search Starts . . . ")
    cv_rf_model = GridSearchCV(estimator=rfc, param_grid=cfg.PARAM_GRID, cv=5)
    print(" Random Forest Parameter Grid Search Completed Fitting Starts . . . ")

    print(" Random Forest Fitting Starts . . . ")
    cv_rf_model.fit(x_train, y_train)
    rf_model = cv_rf_model.best_estimator_
    print(" Random Forest Fitting Complete . . . ")

    # Logistic Regression
    lr_model = LogisticRegression()

    print(" Logisitic Regression Fitting Starts . . . ")
    lr_model.fit(x_train, y_train)
    print(" Logisitic Regression Fitting Complete . . . ")

    print(" Store Trained Models . . . ")
    save_model(rf_model, cfg.MODELS_PATH, cfg.RF_MDL_FILE)
    save_model(lr_model, cfg.MODELS_PATH, cfg.LR_MDL_FILE)
    print(" Store Trained Models Complete. . . ")


    print(" Get model predictions . . . ")
    y_train_preds_rf = rf_model.predict(x_train)
    y_test_preds_rf = rf_model.predict(x_test)

    y_train_preds_lr = lr_model.predict(x_train)
    y_test_preds_lr = lr_model.predict(x_test)
    print(" Get model predictions Complete. . . ")

    print(" Generate Model Performance Reports...")
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    roc_plots([rf_model, lr_model], x_test, y_test)

    feature_importance_plot(rf_model, x_test)
    print(" Generate Model Performance Reports Complete...")
    return rf_model, lr_model



def save_model(model, model_path, model_filename):
    '''
    serialize model to pickle file
    '''
    try:
        assert model is not None
        assert model_path is not None
        assert model_filename is not None
    except AssertionError as excp:
        logging.error(" Missing parameter in calling save_model ")
        raise AssertionError from excp

    joblib.dump(model, os.path.join(model_path, model_filename))



def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder

    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr : test predictions from logistic regression
            y_test_preds_rf : test predictions from random forest

    output: No outputs, writes classification report to
            'random_forest_classification_report.png'
            'logistic_regr_classification_report.png'
    '''

    # Generate report /scores
    plt.rc('figure', figsize=(7, 5))
    fontsize = 11
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': forsize})   ##
    # old approach
    plt.text(0.01, 1.05, str('Random Forest Train'), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.5, str('Random Forest Test'), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(
        os.path.join(
            cfg.RESULTS_PATH,
            'random_forest_classification_report.png'))
    plt.show()

    plt.rc('figure', figsize=(7, 5))
    # approach improved by OP -> monospace!
    plt.text(0.01, 1.05, str('Logistic Regression Train'), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.5, str('Logistic Regression Test'), {
             'fontsize': fontsize}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': fontsize}, fontproperties='monospace')

    plt.axis('off')
    plt.savefig(
        os.path.join(
            cfg.RESULTS_PATH,
            'logistic_regr_classification_report.png'))
    plt.show()


def roc_plots(models, x_data, y_data):
    '''
    produces ROC plot a list of models and stores the resulting plot in the
    in images/results folder

    input:
            models : list of trained models
            X_data:  pandas dataframe of X values
            y_data:  y data corresponding to X values

    output:
            None - output is 'roc_plots.png' written to the path indicated in
                   cfg.results_path
    '''
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    for model in models:
        plot_roc_curve(model, x_data, y_data, ax=axis, alpha=0.8)
    plt.savefig(os.path.join(cfg.RESULTS_PATH, 'roc_plots.png'))
    plt.show()


def feature_importance_plot(model, x_data, output_pth=None):
    '''
    creates and stores the feature importances in pth

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None - Results written as 'feature_importance_plot.png'
                    to path indicated in cfg.results_path
    '''
    if output_pth is None:
        output_pth = cfg.RESULTS_PATH

    # Calculate feature importances
    # importances = cv_rfc.best_estimator_.feature_importances_
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(15, 8))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks([x - 0.5 for x in range(x_data.shape[1])],
               names, rotation=45, fontsize='small')
    plt.tight_layout(pad=1.2)
    plt.savefig(
        os.path.join(
            output_pth,
            'Random_Forest_Feature_Importance_Plot'))
    plt.show()


if __name__ == "__main__":

    dframe_0 = import_data(r"./data/bank_data.csv")
    if dframe_0 is None:
        print(" ERROR: import_data() failed")
    else:
        print(" SUCCESS: import_data() succeeded")

    if perform_eda(dframe_0) == -1:
        print(" ERROR: perform_eda() failed")
        sys.exit(0)
    else:
        print(" SUCCESS: perform_eda() succeeded")

    dframe_1 = encoder_helper(dframe_0, cfg.CAT_COLUMNS)
    if dframe_1 is None :
        print(" ERROR: encoder_helper() failed")
        sys.exit(0)
    else:
        print(" SUCCESS: encoder_helper() succeeded")
        print(dframe_1.head())
