# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Program

## Project Description

In this project with implement two predictive models to determine customer churn based on consumer demogrpahic and 
credit information. The code is implemented using PEP8 guidelines as well on software engineering best practicies 
covered in section one of the ML Ops Nanodegree, clean code principles. 

## Requirements
The following libraries are required for this project:

*   `numpy`
*   `pandas`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `shap`
*   `pylint`
*   `autopep8`

The last two were used for bring the code up to productional level clean code standards.

## Implementation Notes

The code for this project is found in three modules:

*   `churn_config.py`: Contains various parameters used by other modules
*   `churn_library.py`: library containing the churn modules
*   `churn_scripts_logging_and_test.py`: Modules used to test the churn library modules.

All modules have been passed thru autopep and scored at least a 9/10 score on pylint.


## Running Files

*   To run the testing and logging script:

    `python churn_script_logging_and_tests.py`

*   The library routines can be imported and invoked in notebooks, e.g.:

        import churn_library as cls
        . . .
        input_file = os.path.join(cfg.INPUT_PATH, cfg.INPUT_FILE)
        df = cls.import_data(input_file)

</code>


### Outputs:

*   `logs`:  Testing logs are written 
*   `images/eda`: Results of Exploratory data analysis written to this folder
*   `images/results`: Model performance reports and plots are written to this folder


### Note:

There are a couple of refactorization ideas that can be implemented to improve the modularization 
concepts laid out in the course, which I elected not to implement in the interest of time and proceeding 
to the next lesson.

*   The `train_models` module currently trains models, saves the generated models via calls to  `save_model()`, and generates performance 
    reports. One improvement would be to split these out to separate modules. `train_models()` would simply train one or multiple models 
    and return the fitted models (Random Forest, Logistic Regression, etc...) 

*   `classification_report_image()` canbe refactorized to generate reports for one model at a time, instead of receiving 
    ground truth and predictions for multiple models. 