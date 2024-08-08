
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

##### Note: All cd commands assume you are in the project root directory. To access the MLflow UI, visit http://localhost:3000/ in your web browser after the Docker Compose applications are running. 

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

### Run Docker Compose

the following commands will start mage, mlflow, localstack and grafana application which is essential to run this project

`cd app`

`docker-compose up`

### Download Datasets

##### You must download the datasets before running the project, or it won't work. This script fetches six monthly datasets of untouched Chicago taxi data from January 2024 to June 2024. Direct links are not used because the original data is consolidated. Instead, I manually sliced the data by month, uploaded it to my Google Drive, and the script downloads these files to your system and uploads them to the Localstack S3 bucket.

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

### Running Project via Mage

##### Install Packages in Mage

* visit http://localhost:6789/ on your web browser

* in the mage homepage click command center located in top-center position

* open terminal from there and install the following packages

    `pip install mlflow`
    
    `pip install xgboost`

##### Note: You may need to install additional packages not listed above, as system configurations vary. Check error logs and install any required packages accordingly. The code might not run correctly if your system lacks the necessary packages. Even if you've installed some of these packages locally, Mage might not recognize them, so you may need to reinstall them using Mage’s internal terminal..

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

##### Note: You must run the project at least once, either via Docker or Mage, to see relevant information on the Grafana dashboard. Additionally, ensure that the Docker Compose applications are running for Grafana to function correctly.
