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

![data_overview](/starter_file/images/Dataset/data_overview.png)
![data_explore](/starter_file/images/Dataset/data_explore.png)

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
![different_models](/starter_file/images/AutoML/different_models.png)

This could be visualized within the Jupyter Notebook via `RunDetails`:
![run_details](/starter_file/images/AutoML/rundetails.png)

The best model with an accuracy of 87.61 % was VotingEnsemble:
![best_model](/starter_file/images/AutoML/overview_completed.png)

The details of this model could not be shown because of the Udacity lab environment. See the [classroom](https://knowledge.udacity.com/questions/957442?utm_campaign=ret_600_auto_ndxxx_knowledge-answer-created_na&utm_source=blueshift&utm_medium=email&utm_content=ret_600_auto_ndxxx_knowledge-answer-created_na&bsft_clkid=5c8ecb44-1e06-43a2-8961-4c11fee1c69e&bsft_uid=5c9b5725-f4bc-4a88-94a8-2b0fe40ba6e5&bsft_mid=50d4361e-11bc-42f4-881f-6bceb380cb7c&bsft_eid=22b8f7b6-5eac-66ee-cf9f-0d5b86b9fddc&bsft_txnid=aed4cb71-9e4a-4007-b94d-72d9522e365c&bsft_mime_type=html&bsft_ek=2023-02-11T14%3A19%3A04Z&bsft_aaid=8d7e276e-4a10-41b2-8868-423fe96dd6b2&bsft_lx=1&bsft_tv=1#957454) thread here. This is why the model could not be registered in the Jupyter Notebook but manually in the Workplace:
![model_register](/starter_file/images/AutoML/model_register.png)


This model is going to be deployed in a later chapter.

The accuracy could be improved by using different settings, for example by increasing the parameter `experiment_timeout_minutes`. Doing so, AutoML can calculate a greater number of different models and potentially find a new one with a higher accuracy.



## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

As I mentioned above, we are facing a classifaction problem to find out `DEATH_EVENT` of a person. The classification algorithm I used here is Logistic Regression of the `sklearn` package. Logistic regression is a well-known method in statistics that is used to predict the probability of an outcome, and is especially popular for classification tasks.  

The model is included in the [training script] (/starter_file/train.py) and passed to estimator and HyperDrive configurations to predict the best model and accuracy. The HyperDrive run is executed successfully with the help of parameter sampler, policy, estimator. See the `HyperDriveConfig` below: 

```
estimator = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    compute_target=compute_cluster,
    environment=sklearn_env
)

hyperdrive_run_config = HyperDriveConfig(
    run_config=estimator,
    hyperparameter_sampling=param_sampling,
    policy=early_termination_policy,
    primary_metric_name='Accuracy',
    primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=10,
    max_concurrent_runs=4
)
```

Initially in the training script (train.py),the dataset (Heart_Failure_Clinical_Records_Dataset.csv) is retrieved from the URL (https://raw.githubusercontent.com/Harini-Pavithra/Machine-Learning-Engineer-with-Microsoft-Azure-Nanodegree/main/Capstone Project/Dataset/Heart_Failure_Clinical_Records_Dataset.csv) provided using `TabularDatasetFactory` class.Then the data is being split as train and test with the ratio of 70:30.

Via the Hyperparameters of the model I had the control over the training. These Hyperparameters are tuned via `HyperDrive` in order to find the configuration that results in the best performance (here in the highest accuracy). In the Notebook, a parameter sampler of the Class [RandomParameterSampling](https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.randomparametersampling?view=azure-ml-py) was chosen which supports discrete and continuous hyperparameters. It was specified as follows:

```
ps = RandomParameterSampling( {
    "--C" : uniform(0.1,1),
    "--max_iter" : choice(50, 100, 150, 200)
    }
)
```

The [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) says about the parameters:
- `C` (float): Inverse of regularization strength; must be a positive float. That means, we need a continuous variable here, sampled via the `uniform` method.
- `max_iter` (int): Maximum number of iterations taken for the solvers to converge. For this, I used a discrete variable with the `choice` function.

With limiting the range of possible parameters, I was able to shrink the training resources and time.


The [BanditPolicy](https://learn.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.banditpolicy?view=azure-ml-py) defines an early termination policy based on slack criteria, and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor of the evaluation metric with respect to the best performing run will be terminated.
The policy was implemented in the notebook as you can see below:

```
policy = BanditPolicy(
    evaluation_interval=1, 
    slack_factor=0.1,
    delay_evaluation=4
)
```

The policy automatically terminates poorly performing runs and improves computational efficiency so I prevented my training procedure to deal with not promising hyperparameters.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

The best model parameters are
- Max. iterations: 150
- Regularization Strength: 0.8842528
This results in a accuracy of 76.76 %.
![best_model](/starter_file/images/HyperDrive/best_model.png)

This can be visualized in the Notebook via RunDetails:
![run_details](/starter_file/images/HyperDrive/rundetails.png)

The model is registered via Jupyter Notebook:
![register_model](/starter_file/images/HyperDrive/register_model.JPG)

The model can be further improved when tuning other parameters such as the criterion used to define the optimal split. Also, we could consider choosing a different sampler and termination policy or changing their configuration. Furthermore, increasing the `max_total_runs` parameter in the HyperDriveConfig might end up in better results.


## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

AutoML provided a higher accuracy than HyperDrive so the `VotingEnsemble` model is going to be deployed. As already mentioned above, it was not possible to get the output of the best run in the Jupyter Notebook (see [classroom](https://knowledge.udacity.com/questions/957442?utm_campaign=ret_600_auto_ndxxx_knowledge-answer-created_na&utm_source=blueshift&utm_medium=email&utm_content=ret_600_auto_ndxxx_knowledge-answer-created_na&bsft_clkid=5c8ecb44-1e06-43a2-8961-4c11fee1c69e&bsft_uid=5c9b5725-f4bc-4a88-94a8-2b0fe40ba6e5&bsft_mid=50d4361e-11bc-42f4-881f-6bceb380cb7c&bsft_eid=22b8f7b6-5eac-66ee-cf9f-0d5b86b9fddc&bsft_txnid=aed4cb71-9e4a-4007-b94d-72d9522e365c&bsft_mime_type=html&bsft_ek=2023-02-11T14%3A19%3A04Z&bsft_aaid=8d7e276e-4a10-41b2-8868-423fe96dd6b2&bsft_lx=1&bsft_tv=1#957454) thread).

This is why I deployed the model manually in the Azure ML Workspace. The following images shows the enpoint beeing active:

The enpoint can now be consumed via the API:

My sample consumption of the endpoint was done via Jupyter Notebook as you can see below:


As you can see, the sample data would result in a `DEATH_EVENT` of `1`.

In the end, the service and the compute cluster were deleted (manually because of the [classroom](https://knowledge.udacity.com/questions/957442?utm_campaign=ret_600_auto_ndxxx_knowledge-answer-created_na&utm_source=blueshift&utm_medium=email&utm_content=ret_600_auto_ndxxx_knowledge-answer-created_na&bsft_clkid=5c8ecb44-1e06-43a2-8961-4c11fee1c69e&bsft_uid=5c9b5725-f4bc-4a88-94a8-2b0fe40ba6e5&bsft_mid=50d4361e-11bc-42f4-881f-6bceb380cb7c&bsft_eid=22b8f7b6-5eac-66ee-cf9f-0d5b86b9fddc&bsft_txnid=aed4cb71-9e4a-4007-b94d-72d9522e365c&bsft_mime_type=html&bsft_ek=2023-02-11T14%3A19%3A04Z&bsft_aaid=8d7e276e-4a10-41b2-8868-423fe96dd6b2&bsft_lx=1&bsft_tv=1#957454) from above):



## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
