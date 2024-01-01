"""This module provides functions to perform data analysis on customer churn.
It includes functions to load data, perform exploratory data analysis, preprocess data,
train machine learning models, and evaluate their performance.

Author: Jiawei Sun
Creation Date: Dec 31, 2023
"""

# import libraries
import os
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    bank_df = pd.read_csv(pth)
    bank_df['Churn'] = bank_df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return bank_df


def perform_eda(bank_df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    bank_df['Churn'].hist()
    plt.savefig("images/Churn_histogram.png")

    plt.figure(figsize=(20, 10))
    bank_df['Customer_Age'].hist()
    plt.savefig("images/Customer_Age_histogram.png")

    plt.figure(figsize=(20, 10))
    bank_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig("images/Marital_Status_histogram.png")

    plt.figure(figsize=(20, 10))
    sns.histplot(bank_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig("images/Total_Trans_Ct_histogram.png")

    plt.figure(figsize=(20, 10))
    numeric_df = bank_df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig("images/heatmap.png")


def encoder_helper(bank_df, category_lst, response):
    """
    Helper function to turn each categorical column into a new column with the proportion of
    response (e.g., churn) for each category.

    Args:
    df: pandas DataFrame.
    category_lst: List of strings, names of the columns that contain categorical features.
    response: String, name of the response column.

    Returns:
    df: pandas DataFrame with new columns for each categorical feature.
    """
    for category in category_lst:
        encoded_col_name = f"{category}_{response}"
        bank_df[encoded_col_name] = encode_categorical_column(
            bank_df, category, response)
    return bank_df


def encode_categorical_column(bank_df, column, response):
    """
    Encodes a categorical column in the DataFrame to a new column with the mean of
    the response variable for each category.

    Args:
    df: pandas DataFrame.
    column: String, the name of the categorical column to be encoded.
    response: String, the name of the response column.

    Returns:
    A Series with the encoded values.
    """
    group_means = bank_df.groupby(column)[response].mean()
    return bank_df[column].map(group_means)


def perform_feature_engineering(bank_df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for
              naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    y_df = bank_df[response]
    x_df = pd.DataFrame()
    keep_cols = [
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
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    x_df[keep_cols] = bank_df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        x_df, y_df, test_size=0.3, random_state=42)
    return x_train, x_test, y_train, y_test

# pylint: disable=too-many-arguments
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
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Prepare the combined text
    combined_report = (
        "Random Forest Results\n"
        "Test Results\n"
        f"{classification_report(y_test, y_test_preds_rf)}\n"
        "Train Results\n"
        f"{classification_report(y_train, y_train_preds_rf)}\n"
        "Logistic Regression Results\n"
        "Test Results\n"
        f"{classification_report(y_test, y_test_preds_lr)}\n"
        "Train Results\n"
        f"{classification_report(y_train, y_train_preds_lr)}"
    )

    # Classification report
    plt.figure(figsize=(10, 8))  # You may need to adjust the figure size
    plt.text(0.5, 0.5, combined_report, horizontalalignment='center',
             verticalalignment='center', fontsize=10, color='black')
    plt.axis('off')

    plt.savefig(
        'images/combined_classification_report.png',
        bbox_inches='tight')

def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    plt.figure(figsize=(10, 8))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    # Plot
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)

    # Save the figure
    plt.savefig(
        os.path.join(
            output_pth,
            'feature_importance_plot.png'),
        bbox_inches='tight')
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: X training data
              x_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # save best model
    joblib.dump(cv_rfc.best_estimator_, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')

    # ROC
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    # Plot ROC for Linear Regression
    lrc_plot = plot_roc_curve(lrc, x_test, y_test, ax=ax, alpha=0.8)

    # Plot ROC for Random Forest
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, x_test, y_test, ax=ax, alpha=0.8)

    plt.savefig("images/combined_roc.png")

    return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr, y_test_preds_lr


if __name__ == "__main__":
    PATH = r"./data/bank_data.csv"
    df_main = import_data(PATH)
    print(df_main.head())

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
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

    perform_eda(df_main)

    df_main = encoder_helper(df_main, cat_columns, "Churn")

    X_train_main, X_test_main, y_train_main, y_test_main = \
        perform_feature_engineering(df_main, "Churn")

    y_train_preds_rf_main, y_test_preds_rf_main, y_train_preds_lr_main, y_test_preds_lr_main = \
        train_models(X_train_main, X_test_main, y_train_main, y_test_main)

    rfc_model = joblib.load('./models/rfc_model.pkl')
    lr_model = joblib.load('./models/logistic_model.pkl')

    classification_report_image(y_train_main,
                                y_test_main,
                                y_train_preds_lr_main,
                                y_train_preds_rf_main,
                                y_test_preds_lr_main,
                                y_test_preds_rf_main)

    feature_importance_plot(rfc_model, X_test_main, "images/")
