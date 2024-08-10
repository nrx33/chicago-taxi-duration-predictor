

# Chicago Taxi Duration Prediction MLOps Project

### Description
This project focuses on predicting the duration of taxi rides in Chicago. We utilize official taxi ride data from the city to train and evaluate various machine learning models. The process involves selecting the most effective model through training and fine-tuning, and then using it to generate accurate ride duration predictions. By leveraging comprehensive historical data, we aim to enhance the accuracy and reliability of our predictions.

### Technologies
* **Storage:** Localstack S3
* **Experiment Tracking & Model Registry:** mlflow
* **Workflow Orchestration**: Mage
* **Monitoring**: Grafana
* **Containerization**: Docker
* **Best Practices**: Makefile, Unit & Integration Tests, Linter/Code Formatter
* **Deployment**: Batch using containerized code and Mage


# Instructions to run code

### Video Instructions
- [Project Setup](https://www.loom.com/share/5977758cffcc4f9c87c0621718660a3d?sid=1677747c-a03b-40b1-8623-ae8637b67771)
- [Running Application via Docker](https://www.loom.com/share/30b3e4b8879b452eb23487ca62ef67f9?sid=b67644a5-df47-46b2-b72a-1dcd7aad091f)
- [Running Application via Mage](https://www.loom.com/share/f717380f960f41a39dabaa2a6e7d4e74?sid=41d099cd-96e3-44af-911e-829c4d045037)
- [Monitoring Setup](https://www.loom.com/share/e1f2e6f3f20b4b25b5a36f4d1b125411?sid=4729a797-dfab-4183-b023-e0d13ea601fa)
  

### Text Instructions

##### Attention: All `cd` terminal commands assume that you are in the project root directory. Additionally, ensure that your ports are correctly forwarded, as they may not always be automatically forwarded, which can result in the web interfaces of one or more Docker applications being inaccessible. For optimal testing conditions, it is recommended to run this repository in a GitHub Codespace with at least 4 cores of compute power.


### Packages to install in your local environment

    pip install boto3
    pip install gdown
    pip install awscli
    pip install awscli-local
    pip install localstack
    pip install black
    pip install isort
    pip install mlflow
    pip install xgboost
    

##### Note: Your system may require additional packages as every environment setup is different, so read error logs and install those packages as per requirement. The code wont run properly if your system doesn't have the required packages.

### Project Setup
Run these two terminal commands sequentially, it has to be executed before running the docker-compose applications for the first time.

    mkdir -p ./app/resources/grafana \
             ./app/resources/mlflow \
             ./app/resources/mage \
             ./app/resources/localstack

    sudo bash -c "
      chown -R 472:472 ./app/resources/grafana && \
      chown -R 472:472 ./app/resources/mlflow && \
      chown -R 472:472 ./app/resources/mage && \
      chown -R 472:472 ./app/resources/localstack
    "

### Run Docker Compose

the following commands will start mage, mlflow, localstack and grafana application which is essential to run this project

`cd app`

`docker-compose up`

### Download Datasets

##### Note: You must download the datasets before running the project, or it won't work. This script fetches six monthly datasets of untouched Chicago taxi data from January 2024 to June 2024. Direct links are not used because the original data is consolidated. Instead, I manually sliced the data by month, uploaded it to my Google Drive, and the script downloads these files to your system and uploads them to the Localstack S3 bucket.

go to fetch_dataset folder 

`cd app/fetch_dataset`

then run the download_data.py script via terminal

`python download_data.py `

##### Note: You need to make sure the applications from docker-compose are running before running the above terminal command.

After the script is executed successfully you can verify the dataset files in the s3 bucket using the following terminal command

`awslocal s3 ls s3://mlflow-bucket/dataset --recursive`

### Running Project via Docker Container

##### The Docker container runs the entire project workflow, from data loading and transformation to model training, fine-tuning, and registration. It also saves the final predictions. The process dynamically selects one of six datasets per run and utilizes Localstack and MLflow.

in terminal run the following code 

`cd app`

`make run`

##### Note: You need to make sure the applications from docker-compose are running before running the above terminal command.

After the script is executed successfully you can verify the output file in the s3 bucket using the following terminal command

`awslocal s3 ls s3://mlflow-bucket/output --recursive`

##### a successful run log should look something like this

    Starting main process...
    Setting environment variables...
    Bucket 'mlflow-bucket' already exists.
    Selected month: 3
    Reading data from S3...
    Preprocessing data...
    Preparing features...
    Splitting data into training and testing sets...
    Training models...
    Training models: 100%|████████████████████████████████████████████████████████████████████████████████| 4/4 [05:32<00:00, 83.11s/it]
    Best model: XGBoost
    Tuning hyperparameters for the best model: XGBoost...
    Best parameters: {'n_estimators': 50, 'max_depth': 7, 'learning_rate': 0.2}
    Registering the best model: XGBoost...
    Registered model 'XGBoost' already exists. Creating a new version of this model...
    2024/08/08 12:15:12 INFO mlflow.store.model_registry.abstract_store: Waiting up to 300 seconds for model version to finish creation. Model name: XGBoost, version 10
    Created version '10' of model 'XGBoost'.
    Tuned XGBoost MAE: 5.784219741284125, RMSE: 8.147295241948855, R²: 0.5810439060437554
    Run ID of the best-tuned model: 122827ea0bad4806b655734b50e6493f
    Saving predictions to S3...
    Output CSV file 'Taxi_Trips_2024_03_predictions_20240808_121518.csv' has been saved to the S3 bucket.

### Running Project via Mage

##### Install Packages in Mage

* visit http://localhost:6789/ on your web browser

* in the mage homepage click command center located in top-center position

* search and open terminal from there and install the following packages

    `pip install mlflow`
    
    `pip install xgboost`

##### Note: You may need to install additional packages not listed above, as system configurations vary. Check error logs and install any required packages accordingly. The code might not run correctly if your system lacks the necessary packages. Even if you've installed some of these packages locally, Mage might not recognize them, so you may need to reinstall them using Mage’s internal terminal. And just like the previous section the data is dynamically chosen between the six datasets.

##### Import Existing Pipeline

* download or copy the train_batch.zip from `app/mage_backup` folder to your host machine

* visit http://localhost:6789/ on your web browser

* on the left sidebar in mage click on pipelines

* click on New 

* then click import pipeline zip

* upload the train_batch.zip you downloaded earlier 

* after it says successful import click on close

##### Run the Pipeline

* visit http://localhost:6789/ on your web browser

* on the left sidebar in mage click on pipelines

* now click on the train_batch pipeline 

* then on the next window click Run@once

* then a popup will come just click Run now

##### Note: You need to make sure the applications from docker-compose are running for the above instructions to work.

![Mage Trigger Run](https://github.com/nrx33/taxi_chicago_prediction_mlops/raw/main/assets/mage_trigger_runs.png)

### Experiment tracking and model registry using mlflow

#####  To access the MLflow UI, visit http://localhost:5000/ in your web browser.

![MLflow Experiment](https://github.com/nrx33/taxi_chicago_prediction_mlops/raw/main/assets/mlflow_experiment.png)

##### Note: You need to make sure the applications from docker-compose are running for mlflow to work.

![MLflow Model Registry](https://github.com/nrx33/taxi_chicago_prediction_mlops/raw/main/assets/mlflow_model_registry.png)

### Monitoring using Grafana
##### Adding Data Source
* visit http://localhost:3000/ on your web browser
* both username and password is "admin"
* after initial login it will give you an option to change password, you can either change or skip it
* Click on add new connection on left sidebar
* search for sqlite and then click on it
* in the sqlite page click on install on top right and wait until its installed
* then click on add new data source on top right
* in the path textbox type
`/mlflow/mlflow.db`

![Grafana Connection Setup](https://github.com/nrx33/taxi_chicago_prediction_mlops/raw/main/assets/grafana_db_setup.png)


* then click save and test
* then if it says Data source is working then you have successfully added the data source

##### Importing Dashboard
* visit http://localhost:3000/ on your web browser
* click on dashboards in the left sidebar
* click on create dashboard
* in the next window click import a dashboard
* then you can either upload the dashboard.json or copy the json code from dashboard.json file and paste it on the provided textbox. You can find the dashboard.json file in the `app/grafana_backup` folder.
* then click on load only if you copied the code and pasted in the textbox, if you uploaded the file instead you can skip this part
* in the next window click on import
* if all is done correctly you will be redirected to the dashboard.

##### Note: You must run the project successfully at least once, either via Docker or Mage, to see relevant information on the Grafana dashboard. Additionally, ensure that the Docker Compose applications are running for Grafana to function correctly. The run status defaults to successful run in the dashboard so if there are no failed runs it will only show the successful runs until a failed run is detected.

![Grafana Dashboard](https://github.com/nrx33/taxi_chicago_prediction_mlops/raw/main/assets/grafana_dashboard.png)
