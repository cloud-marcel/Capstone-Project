*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Capstone Project: Heart Failure Prediction

This final project will finish my Udacity Azure ML Nanodegree. I need to use the knowledge I have obtained from the course to solve an interesting problem. In this project, you am going to create two models: one using AutoML and one customized model whose hyperparameters are tuned using HyperDrive. I will then compare the performance of both the models and deploy the best performing model.

To achieve this, I will import an external dataset into my workspace, train a model using the different tools available in the AzureML framework as well as deploy the model as a web service using Azure Container Instances(ACI). Once the model is deployed, the endpoint will be consumed.


## Project Set Up and Installation
For this project, the Lab provided by Udacity was used. The workspace configuration was the following [config.json] ###### ToDo

## Dataset

### Overview
In this project, a publicly available dataset [Heart Failure Prediction dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) was used which was recommended by the instructor.
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
The last column of the dataset is the label `DEATH_EVENT` to be predicted.

### Task
Training the two models, I am going to predict if a person with CVD having specific features (like diabetes, sex, age) will survive the disease or not. This represents a classifcation problem.

### Access
The dataset was downloaded from the Kaggle website and saved locally. It was then added to the workspace to work with:

####### ToDo: Bild von Dataset

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

AutoML is a new feature of the Azure ML Workspace to automate the time consuming, iterative tasks of machine learning model development. In contrast to Hyperparameter tuning  with HyperDrive (see next chapter), you don't need a model which is specified by the ML engineer before the training. Rather AutoML finds a model by using different algorithms and parameters trying to improve the specified metrics.

The orchestration is done in a separate Notebook. I do not need a training script here. The applied settings are shown below. They are passed to the AutoMLConfig object in JSON format. As already mentioned above, the column `DEATH_EVENT` is going to be predicted.

```
automl_settings = {
    "experiment_timeout_minutes":25,
    "task":'classification',
    "primary_metric":'accuracy',
    "training_data":dataset,
    "label_column_name":'DEATH_EVENT',
    "n_cross_validations":4,
    "max_concurrent_iterations": 4,
    "featurization": 'auto'
}

automl_config = AutoMLConfig(
    compute_target=cpu_cluster,
    **automl_settings
)
```

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

Several models using different algorithms were found automatically by AutoML, e. g. LightGBM, RandomForest, ...
######## ToDo: different models

This could be visualized within the Jupyter Notebook via `RunDetails`:
######## ToDo: Run Details

The best model with an accuracy of ##### was VotingEnsemble with the specific paramters:
######## ToDo: Detailed View with parameters

This model is going to be deployed in a later chapter.

The accuracy could be improved by using different settings, for example by increasing the parameter `experiment_timeout_minutes`. Doing so, AutoML can calculate a greater number of different models and potentially find a new one with a higher accuracy.



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
