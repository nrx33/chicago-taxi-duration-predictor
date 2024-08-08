

### Chicago Taxi Duration Prediction MLOPS Project

### Packages to install in your local environment

    pip install boto3
    pip install gdown
    pip install awscli
    pip install awscli-local
    pip install localstack
    pip install isort
    pip install mlflow
    pip install xgboost

##### Note: Your system may require additional packages as every environment setup is different, so read error logs and install those packages as per requirement. The code wont run properly if your system doesn't have the required packages.

### Project Setup
run these three commands on the terminal one by one sequentially

    mkdir -p ./app/resources/grafana \
             ./app/resources/mlflow \
             ./app/resources/mage \
             ./app/resources/localstack

`sudo chown -R 472:472 ./resources/grafana`

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

##### Note: You might even have to install packages not listed above as the configuration for every system is different, so read error logs and install those packages as per requirement. The code wont run properly if your system doesn't have the required packages. Even, if you already installed some of these packages in your local environment it may not be recognized inside mage so you might have to install some packages again for mage using's mage internal terminal.

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