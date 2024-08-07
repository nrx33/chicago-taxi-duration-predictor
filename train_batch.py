"""
This script handles the process of training multiple regression models
for predicting taxi trip duration. It includes data preprocessing,
feature preparation, model training, hyperparameter tuning, and saving
predictions to S3.
"""

# Import necessary libraries and modules
import os
import random
import time
import uuid
import warnings

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from xgboost import XGBRegressor


def set_environment(bucket_name):
    """
    Set environment variables for MLflow and S3.

    Args:
        bucket_name (str): Name of the S3 bucket to be used.

    Returns:
        boto3.client: Initialized S3 client.
    """
    print("Setting environment variables...")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localstack:4566"
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"
    mlflow.set_tracking_uri("http://mlflow:5000")

    # Initialize boto3 client for S3
    s3_client = boto3.client("s3", endpoint_url="http://localstack:4566")

    # Create the bucket if it does not exist
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' already exists.")
    except boto3.exceptions.Boto3Error:
        print(f"Bucket '{bucket_name}' does not exist. Creating...")
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"Bucket '{bucket_name}' created successfully.")

    return s3_client


def read_data(file_source):
    """
    Read data from a given source file.

    Args:
        file_source (str): Source file path to read data from.

    Returns:
        pandas.DataFrame: DataFrame containing the read data.
    """
    print("Reading data from S3...")
    return pd.read_csv(
        file_source,
        storage_options={
            "key": "test",
            "secret": "test",
            "client_kwargs": {"endpoint_url": "http://localstack:4566"},
        },
    )


def preprocess_data(df):
    """
    Preprocess the data for training.

    Args:
        df (pandas.DataFrame): DataFrame containing raw data.

    Returns:
        pandas.DataFrame: DataFrame containing preprocessed data.
    """
    print("Preprocessing data...")
    # Drop rows with missing values in specific columns
    df = df.dropna(
        subset=[
            "Trip Miles",
            "Trip Start Timestamp",
            "Trip End Timestamp",
            "Payment Type",
            "Company",
            "Duration",
        ]
    ).copy()

    # Convert timestamps to datetime
    df["Trip Start Timestamp"] = pd.to_datetime(
        df["Trip Start Timestamp"], errors="coerce"
    )
    df["Trip End Timestamp"] = pd.to_datetime(df["Trip End Timestamp"], errors="coerce")

    # Drop rows where datetime conversion failed
    df = df.dropna(subset=["Trip Start Timestamp", "Trip End Timestamp"]).copy()

    # Extract hour and day of the week from timestamps
    df["Trip Start Hour"] = df["Trip Start Timestamp"].dt.hour
    df["Trip Start DayOfWeek"] = df["Trip Start Timestamp"].dt.dayofweek
    df["Trip End Hour"] = df["Trip End Timestamp"].dt.hour
    df["Trip End DayOfWeek"] = df["Trip End Timestamp"].dt.dayofweek

    return df


def prepare_features(df):
    """
    Prepare features for model training.

    Args:
        df (pandas.DataFrame): DataFrame containing preprocessed data.

    Returns:
        tuple: Tuple containing encoded features and target variable.
    """
    print("Preparing features...")
    # Select relevant features and target variable
    features = df[
        [
            "Trip Miles",
            "Trip Start Hour",
            "Trip Start DayOfWeek",
            "Trip End Hour",
            "Trip End DayOfWeek",
            "Payment Type",
            "Company",
        ]
    ]
    target = df["Duration"]
    # One-hot encode categorical features
    encoder = OneHotEncoder(drop="first", sparse_output=False)
    encoded_features = pd.DataFrame(
        encoder.fit_transform(features[["Payment Type", "Company"]]),
        columns=encoder.get_feature_names_out(),
    )
    # Combine numerical and encoded categorical features
    numerical_features = features.drop(columns=["Payment Type", "Company"])
    return (
        pd.concat(
            [
                numerical_features.reset_index(drop=True),
                encoded_features.reset_index(drop=True),
            ],
            axis=1,
        ),
        target,
    )


def split_data(features_encoded, target):
    """
    Split data into training and testing sets.

    Args:
        features_encoded (pandas.DataFrame): DataFrame containing encoded features.
        target (pandas.Series): Series containing target variable.

    Returns:
        tuple: Tuple containing training and testing sets for features and target.
    """
    print("Splitting data into training and testing sets...")
    return train_test_split(features_encoded, target, test_size=0.2, random_state=42)


def train_models(x_train, y_train, x_test, y_test):
    """
    Train multiple models and log results with MLflow.

    Args:
        x_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target variable.
        x_test (pandas.DataFrame): Testing features.
        y_test (pandas.Series): Testing target variable.

    Returns:
        tuple: Tuple containing the name and instance of the best model.
    """
    print("Training models...")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
    }

    mlflow.set_experiment("Taxi Trip Duration Prediction")
    best_model, best_model_name, best_metrics = (
        None,
        None,
        (float("inf"), float("inf"), float("-inf")),
    )

    pbar = tqdm(models.items(), desc="Training models")
    for name, model in pbar:
        start_time = time.time()
        with mlflow.start_run(run_name=name):
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)
            metrics = (
                mean_absolute_error(y_test, y_pred),
                mean_squared_error(y_test, y_pred, squared=False),
                r2_score(y_test, y_pred),
            )
            log_metrics_and_model(name, model, metrics, start_time)
            if is_best_model(metrics, best_metrics):
                best_model, best_model_name, best_metrics = model, name, metrics
    return best_model_name, best_model


def log_metrics_and_model(name, model, metrics, start_time):
    """
    Log model metrics and instance with MLflow.

    Args:
        name (str): Name of the model.
        model (object): Trained model instance.
        metrics (tuple): Tuple containing MAE, RMSE, and R² metrics.
        start_time (float): Time when training started.
    """
    mlflow.log_param("model", name)
    mlflow.log_metrics(
        {
            "mae": metrics[0],
            "RMSE": metrics[1],
            "R2": metrics[2],
            "elapsed_time": time.time() - start_time,
        }
    )
    mlflow.sklearn.log_model(model, "model")


def is_best_model(metrics, best_metrics):
    """
    Determine if the current model is the best model.

    Args:
        metrics (tuple): Tuple containing current model metrics.
        best_metrics (tuple): Tuple containing best model metrics.

    Returns:
        bool: True if current model is better than the best model, False otherwise.
    """
    return (
        metrics[0] < best_metrics[0]
        and metrics[1] < best_metrics[1]
        and metrics[2] > best_metrics[2]
    )


def tune_hyperparameters(best_model_name, best_model, x_train, y_train):
    """
    Tune hyperparameters for the best model.

    Args:
        best_model_name (str): Name of the best model.
        best_model (object): Instance of the best model.
        x_train (pandas.DataFrame): Training features.
        y_train (pandas.Series): Training target variable.

    Returns:
        tuple: Tuple containing the best estimator and its parameters.
    """
    print(f"Tuning hyperparameters for the best model: {best_model_name}...")
    param_distributions = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
        },
        "Linear Regression": {
            "fit_intercept": [True, False],
            "normalize": [True, False],
        },
    }.get(best_model_name, {})

    random_search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_distributions,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_iter=10,
        n_jobs=-1,
    )
    random_search.fit(x_train, y_train)
    return random_search.best_estimator_, random_search.best_params_


def register_best_model(best_model_name, best_model, x_test, y_test):
    """
    Register the best model with MLflow.

    Args:
        best_model_name (str): Name of the best model.
        best_model (object): Instance of the best model.
        x_test (pandas.DataFrame): Testing features.
        y_test (pandas.Series): Testing target variable.

    Returns:
        str: Run ID of the registered best model.
    """
    print(f"Registering the best model: {best_model_name}...")
    with mlflow.start_run(run_name=f"Tuned {best_model_name}"):
        y_pred = best_model.predict(x_test)
        metrics = (
            mean_absolute_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred, squared=False),
            r2_score(y_test, y_pred),
        )
        mlflow.log_param("model", best_model_name)
        mlflow.log_metrics({"mae": metrics[0], "RMSE": metrics[1], "R2": metrics[2]})
        mlflow.sklearn.log_model(best_model, "model")
        run_id = mlflow.active_run().info.run_id
        mlflow.register_model(f"runs:/{run_id}/model", best_model_name)
        print(
            f"Tuned {best_model_name} MAE: {metrics[0]}, RMSE: {metrics[1]}, R²: {metrics[2]}"
        )
        print(f"Run ID of the best-tuned model: {run_id}")
        return run_id


def generate_uuid(n):
    """
    Generate a list of UUIDs.

    Args:
        n (int): Number of UUIDs to generate.

    Returns:
        list: List of generated UUID strings.
    """
    return [str(uuid.uuid4()) for _ in range(n)]


def save_predictions_to_s3(df, features_encoded, best_model, run_id, file_source):
    """
    Save predictions to S3.

    Args:
        df (pandas.DataFrame): Original DataFrame containing data.
        features_encoded (pandas.DataFrame): DataFrame containing encoded features.
        best_model (object): Instance of the best model.
        run_id (str): Run ID of the registered best model.
        file_source (str): Source file path of the data.
    """
    print("Saving predictions to S3...")
    original_filename = os.path.basename(file_source)
    filename_without_extension = os.path.splitext(original_filename)[0]
    y_pred_all = best_model.predict(features_encoded)

    # Create an output DataFrame with relevant columns and predictions
    output_df = df[
        [
            "Trip Miles",
            "Trip Start Timestamp",
            "Trip End Timestamp",
            "Payment Type",
            "Company",
            "Duration",
        ]
    ].copy()
    output_df["Predicted Duration"] = np.round(y_pred_all, 2)
    output_df.rename(columns={"Duration": "Actual Duration"}, inplace=True)
    output_df["Ride ID"] = generate_uuid(len(output_df))
    output_df["Difference"] = round(
        output_df["Actual Duration"] - output_df["Predicted Duration"], 2
    )
    output_df["Model Version"] = run_id
    output_df = output_df[
        [
            "Ride ID",
            "Trip Miles",
            "Trip Start Timestamp",
            "Trip End Timestamp",
            "Payment Type",
            "Company",
            "Actual Duration",
            "Predicted Duration",
            "Difference",
            "Model Version",
        ]
    ]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{filename_without_extension}_predictions_{timestamp}.csv"
    s3_path = f"s3://mlflow-bucket/output/{output_filename}"
    output_df.to_csv(
        s3_path,
        index=False,
        storage_options={
            "key": "test",
            "secret": "test",
            "client_kwargs": {"endpoint_url": "http://localstack:4566"},
        },
    )
    print(f"Output CSV file '{output_filename}' has been saved to the S3 bucket.")


def get_random_month():
    """
    Randomly select a month for data file.

    Returns:
        int: Randomly selected month (1 to 6).
    """
    return random.randint(1, 6)


def main():
    """
    Execute the entire workflow.

    This function sets up the environment, reads and preprocesses data,
    trains and evaluates models, and saves the results.
    """
    print("Starting main process...")
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings(
        "ignore", message="'squared' is deprecated", category=FutureWarning
    )
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

    # Set the bucket name
    bucket_name = "mlflow-bucket"

    # Set the environment and create the bucket if necessary
    _ = set_environment(bucket_name)

    # Randomly select a month
    month = get_random_month()
    print(f"Selected month: {month}")

    # Format the filename with the selected month
    file_name = f"Taxi_Trips_2024_{month:02d}.csv"
    file_source = f"s3://mlflow-bucket/dataset/{file_name}"

    df = read_data(file_source)
    df = preprocess_data(df)
    features_encoded, target = prepare_features(df)
    x_train, x_test, y_train, y_test = split_data(features_encoded, target)

    best_model_name, best_model = train_models(x_train, y_train, x_test, y_test)
    print(f"Best model: {best_model_name}")

    best_model, best_params = tune_hyperparameters(
        best_model_name, best_model, x_train, y_train
    )
    print(f"Best parameters: {best_params}")

    run_id = register_best_model(best_model_name, best_model, x_test, y_test)
    save_predictions_to_s3(df, features_encoded, best_model, run_id, file_source)


# Entry point of the script
if __name__ == "__main__":
    main()
