"""This module provides functions to perform data analysis on customer churn.
It includes functions to load data, perform exploratory data analysis, preprocess data,
train machine learning models, and evaluate their performance.

Author: Jiawei Sun
Creation Date: Dec 31, 2023
"""

# import libraries
import os
import logging
import numpy as np
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        bank_df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert bank_df.shape[0] > 0
        assert bank_df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return bank_df


def test_eda(perform_eda, bank_df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(bank_df)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: Error occurred during EDA")
        raise err

    # List of expected files
    expected_files = [
        "images/Churn_histogram.png",
        "images/Customer_Age_histogram.png",
        "images/Marital_Status_histogram.png",
        "images/Total_Trans_Ct_histogram.png",
        "images/heatmap.png"
    ]

    try:
        for file in expected_files:
            assert os.path.isfile(file), f"File not created: {file}"
        logging.info(
            "Testing perform_eda: All EDA images are successfully created")
    except AssertionError as err:
        logging.error("Testing perform_eda: Some EDA images are missing")
        raise err


def test_encoder_helper(encoder_helper, bank_df):
    '''
    test encoder helper
    '''

    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    try:
        # Apply the encoder_helper function
        df_encoded = encoder_helper(bank_df, category_lst, 'Churn')
        logging.info("Testing encoder_helper: SUCCESS")

        # Check for the creation of new columns
        for category in category_lst:
            encoded_col_name = f"{category}_Churn"
            assert encoded_col_name in df_encoded.columns, f"Column {encoded_col_name} not created"
            assert df_encoded[encoded_col_name].isnull().sum(
            ) == 0, f"Column {encoded_col_name} contains NaN values"
            assert df_encoded[
                encoded_col_name].dtype == np.float64,\
                f"Column {encoded_col_name} is not of type float64"

        logging.info(
            "Testing encoder_helper: New columns correctly created and encoded")

    except Exception as err:
        logging.error("Testing encoder_helper: Error occurred")
        raise err


def test_perform_feature_engineering(perform_feature_engineering, bank_df):
    '''
    test perform_feature_engineering
    '''

    try:
        # Apply the perform_feature_engineering function
        x_train, x_test, y_train, y_test = perform_feature_engineering(
            bank_df, 'Churn')
        logging.info("Testing perform_feature_engineering: SUCCESS")

        # Check the shape of the outputs
        assert x_train.shape[0] > 0, "X_train has no rows"
        assert x_test.shape[0] > 0, "X_test has no rows"
        assert y_train.shape[0] > 0, "y_train has no rows"
        assert y_test.shape[0] > 0, "y_test has no rows"

        # Check if the response variable is dropped from X
        assert 'Churn' not in x_train.columns, "'Churn' should not be in X_train"
        assert 'Churn' not in x_test.columns, "'Churn' should not be in X_test"

        # Check for correct split ratio (approximately 70-30 split)
        total_rows = bank_df.shape[0]
        assert abs(
            x_train.shape[0] - 0.7 * total_rows) < 10, "Incorrect train-test split ratio for X"
        assert abs(
            x_test.shape[0] - 0.3 * total_rows) < 10, "Incorrect train-test split ratio for X"
        assert abs(
            y_train.shape[0] - 0.7 * total_rows) < 10, "Incorrect train-test split ratio for y"
        assert abs(
            y_test.shape[0] - 0.3 * total_rows) < 10, "Incorrect train-test split ratio for y"

        logging.info(
            "Testing perform_feature_engineering: Data split correctly")

    except Exception as err:
        logging.error("Testing perform_feature_engineering: Error occurred")
        raise err


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''

    try:
        # Apply the train_models function
        y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr = train_models(
            x_train, x_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")

        # Check if the models are saved
        assert os.path.isfile(
            'models/rfc_model.pkl'), "Random Forest model file not found"
        assert os.path.isfile(
            'models/logistic_model.pkl'), "Logistic Regression model file not found"

        # Check the predictions
        assert len(y_train_preds_rf) == len(
            y_train), "Mismatch in number of predictions for Random Forest (train)"
        assert len(y_test_preds_rf) == len(
            y_test), "Mismatch in number of predictions for Random Forest (test)"
        assert len(y_train_preds_lr) == len(
            y_train), "Mismatch in number of predictions for Logistic Regression (train)"
        assert len(y_test_preds_lr) == len(
            y_test), "Mismatch in number of predictions for Logistic Regression (test)"

        logging.info(
            "Testing train_models: Model predictions and saving are correct")

    except Exception as err:
        logging.error("Testing train_models: Error occurred")
        raise err


if __name__ == "__main__":
    # Run test functions
    bank_df_sample = test_import(cls.import_data)

    # Pass the DataFrame to other test functions as needed
    test_eda(cls.perform_eda, bank_df_sample)
    test_encoder_helper(cls.encoder_helper, bank_df_sample)
    test_perform_feature_engineering(cls.perform_feature_engineering, bank_df_sample)

    # Split the data for train_models
    X_train_sample, X_test_sample, y_train_sample, y_test_sample = cls.perform_feature_engineering(
        bank_df_sample, 'Churn')
    test_train_models(cls.train_models,
                      X_train_sample, X_test_sample, y_train_sample, y_test_sample)
