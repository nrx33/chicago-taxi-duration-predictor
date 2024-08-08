import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from train_batch import (
    preprocess_data,
    prepare_features,
    split_data,
    train_models,
    tune_hyperparameters,
    generate_uuid
)

@pytest.fixture
def sample_data():
    # Function to generate synthetic data
    def generate_data(num_samples):
        data = {
            "Trip ID": [f"trip_{i}" for i in range(num_samples)],
            "Taxi ID": [f"taxi_{i}" for i in range(num_samples)],
            "Trip Start Timestamp": pd.date_range("2024-01-01", periods=num_samples, freq="h"),
            "Trip End Timestamp": pd.date_range("2024-01-01 01:00:00", periods=num_samples, freq="h"),
            "Trip Miles": [round(5 + i * 0.5, 2) for i in range(num_samples)],
            "Pickup Census Tract": [17031980000.0 if i % 2 == 0 else None for i in range(num_samples)],
            "Dropoff Census Tract": [17031320100.0 if i % 2 == 0 else None for i in range(num_samples)],
            "Pickup Community Area": [76.0 if i % 2 == 0 else 6.0 for i in range(num_samples)],
            "Dropoff Community Area": [32.0 if i % 2 == 0 else None for i in range(num_samples)],
            "Fare": [round(10 + i * 1.5, 2) for i in range(num_samples)],
            "Tips": [2.0 if i % 2 == 0 else 0.0 for i in range(num_samples)],
            "Tolls": [0.0 for _ in range(num_samples)],
            "Extras": [1.0 if i % 2 == 0 else 0.0 for i in range(num_samples)],
            "Trip Total": [round(15 + i * 1.5, 2) for i in range(num_samples)],
            "Payment Type": ["Credit Card" if i % 2 == 0 else "Cash" for i in range(num_samples)],
            "Company": ["Flash Cab" if i % 2 == 0 else "Unknown" for i in range(num_samples)],
            "Pickup Centroid Latitude": [41.97907082 for _ in range(num_samples)],
            "Pickup Centroid Longitude": [-87.903039661 for _ in range(num_samples)],
            "Pickup Centroid Location": ["POINT (-87.9030396611 41.9790708201)" for _ in range(num_samples)],
            "Dropoff Centroid Latitude": [41.884987192 for _ in range(num_samples)],
            "Dropoff Centroid Longitude": [-87.620992913 for _ in range(num_samples)],
            "Dropoff Centroid  Location": ["POINT (-87.6209929134 41.8849871918)" for _ in range(num_samples)],
            "Duration": [60.0 if i % 2 == 0 else 30.0 for i in range(num_samples)]
        }
        return pd.DataFrame(data)

    # Generate a larger dataset
    return generate_data(100)

@patch('train_batch.mlflow.start_run')
@patch('train_batch.mlflow.sklearn.log_model')
@patch('train_batch.mlflow.log_metrics')
@patch('train_batch.mlflow.log_param')
@patch('train_batch.RandomizedSearchCV')  # Mock RandomizedSearchCV
def test_integration(mock_random_search_cv, mock_log_param, mock_log_metrics, mock_log_model, mock_start_run, sample_data):
    # Mock the result of RandomizedSearchCV to return the model without changing parameters
    mock_search_instance = MagicMock()
    mock_search_instance.best_estimator_ = LinearRegression()
    mock_search_instance.best_params_ = {"fit_intercept": True, "positive": False}
    mock_random_search_cv.return_value = mock_search_instance
    
    # Load the sample data
    df = sample_data
    
    # Run the workflow steps
    df = preprocess_data(df)
    features_encoded, target = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(features_encoded, target)
    
    # Verify that each model runs in its own distinct run
    mock_start_run.reset_mock()
    best_model_name, best_model = train_models(X_train, y_train, X_test, y_test)
    assert best_model_name is not None, "A best model should be selected."
    
    best_model, best_params = tune_hyperparameters(best_model_name, best_model, X_train, y_train)
    assert best_model is not None, "Hyperparameter tuning should produce a valid model."

    # Fit the best model with training data for prediction
    best_model.fit(X_train, y_train)

    # Mock the UUID generation to have predictable results
    with patch('train_batch.generate_uuid', return_value=['uuid1', 'uuid2']):
        # Simulate making predictions
        predictions = best_model.predict(features_encoded)
        assert len(predictions) == len(target), "Predictions should match the number of input samples."

    # Validate MLflow calls
    assert mock_start_run.call_count == 4, "Each model should have its own run"
    mock_log_model.assert_called()
    mock_log_metrics.assert_called()
    mock_log_param.assert_called()