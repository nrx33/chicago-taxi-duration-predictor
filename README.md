# Chicago Taxi Duration Prediction MLOps Project

### Description
This project aims to predict the ride duration of taxi rides in Chicago. Using official taxi ride data from the city of Chicago we train machine learning models and find the best model for the dataset, tune it and then final predictions are made.

### Technologies
**Storage:** Localstack S3

**Experiment Tracking & Model Registry:** mlflow

**Workflow Orchestration**: Mage

**Monitoring**: Grafana

**Containerization**: Docker

# Instructions to run code

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
run these two terminal commands sequentially

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

the following commands will start mage, mlflow, localstack and grafana application

`cd app`

`docker-compose up`

### Download Datasets

##### You need to download the datasets before running the project or it wont work. This script will download six datasets from January 2024 to June 2024 untouched taxi data from city of Chicago, direct links from the original website are not used because the data is consolidated and is not separated by month, i manually sliced the data into specific months and uploaded to my personal google drive and these files are then downloaded to your system and uploaded to the localstack s3 bucket via the provided script. 

go to fetch_dataset folder 

`cd app/fetch_dataset`

then run the download_data.py script via terminal

`python download_data.py `

##### Note: You need to make sure the applications from docker-compose are running before running the above terminal command.

After the script is executed successfully you can verify the dataset files in the s3 bucket using the following terminal command

`awslocal s3 ls s3://mlflow-bucket/dataset --recursive`

### Running Project via Docker Container

##### The application in the docker container will run the project from loading the data to transforming the data to training the models to finetuning the best performing model then registering the best model and finally saving the output with predictions. Since there are six datasets it is dynamically chosen in every run which one is used. The whole process uses localstack and mlflow. 

in terminal run the following code 

`cd app`

`make run`

##### Note: You need to make sure the applications from docker-compose are running before running the above terminal command.

After the script is executed successfully you can verify the output file in the s3 bucket using the following terminal command

`awslocal s3 ls s3://mlflow-bucket/output --recursive`

### Running Project via Mage

##### Install Packages in Mage

* visit http://localhost:6789/ on your web browser

* in the mage homepage click command center located in top-center position

* open terminal from there and install the following packages

    `pip install mlflow`
    
    `pip install xgboost`

##### Note: You might even have to install packages not listed above as the configuration for every system is different, so read error logs and install those packages as per requirement. The code wont run properly if your system doesn't have the required packages. Even, if you already installed some of these packages in your local environment it may not be recognized inside mage so you might have to install some packages again using mage's internal terminal.

##### Import Existing Pipeline

* download or copy the train_batch.zip from mage_backup folder to your host machine

* visit http://localhost:6789/ on your web browser

* on the left sidebar in mage click on pipelines

* click on New 

* then click import pipeline zip

* upload the train_batch.zip you downloaded earlier 

* After it says successful import click on close

##### Run the Pipeline

* visit http://localhost:6789/ on your web browser

* on the left sidebar in mage click on pipelines

* now click on the train_batch pipeline 

* then on the next window click Run@once

* then a popup will come just click Run now

##### Note: You need to make sure the applications from docker-compose are running for the above instructions to work.

### Monitoring with Grafana
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
* Then click save and test
* Then if it says Data source is working then you have successfully added the data source

##### Importing Dashboard
* visit http://localhost:3000/ on your web browser
* click on dashboards in the left sidebar
* click on create dashboard
* in the next window click import dashboard
* then you can either upload the dashboard.json or copy the json code from dashboard.json file and paste it on the provided textbox. You can find the dashboard.json file in the `app/grafana_backup` folder.
* then click on load
* in the next window click on import
* if all is done correctly you will be redirected to the dashboard.

##### Note: You need to at least run the project at least once either via docker or mage to be able to see relevant information on the grafana dashboard. Also, to make grafana work you need to also make sure the docker-compose applications are running.
