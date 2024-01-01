# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project, I will implement my learnings to identify credit card customers that are most likely to churn. It will require the skills for testing, logging, and best coding practices from this lesson. The completed project will include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Files and data description
The goal is to refactor the given churn_notebook.ipynb file following the best coding practices to generate these files:
- churn_library.py
- churn_script_logging_and_tests.py
- README.md

## Running Files

#### Libraries Required
* scikit-learn==0.24.1
* shap==0.40.0
* joblib==1.0.1
* pandas==1.2.4
* numpy==1.20.1
* matplotlib==3.3.4
* seaborn==0.11.2
* pylint==2.7.4
* autopep8==1.5.6

#### Code Quality
- Format the refactored code using PEP 8 – Style Guide.
`autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py`

- Use Pylint for the code analysis looking for programming errors, and scope for further refactoring. 
`pylint churn_library.py
pylint churn_script_logging_and_tests.py`


#### churn_library.py

The churn_library.py is a library of functions to find customers who are likely to churn. You may choose to add an if __name__ == "__main__" block that allows you to run the code below and understand the results for each of the functions and refactored code associated with the original notebook.

`ipython churn_library.py`


#### churn_script_logging_and_tests.py

This file should:
1. Contain unit tests for the churn_library.py functions. Use the basic assert statements that test functions work properly. The goal of test functions is to checking the returned items aren't empty or folders where results should land have results after the function has been run.

2. Log any errors and INFO messages. You should log the info messages and errors in a .log file, so it can be viewed post the run of the script. The log messages should easily be understood and traceable. Also, ensure that testing and logging can be completed on the command line, meaning, running the below code in the terminal should test each of the functions and provide any errors to a file stored in the /logs folder.
`ipython churn_script_logging_and_tests.py`
Testing framework: As long as you fulfil all the rubric criteria, the choice of testing framework rests with the student. For instance, you can use pytest for writing functional tests.
