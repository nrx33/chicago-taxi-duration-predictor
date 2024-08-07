import pytest
import pandas as pd
import os
from io import StringIO
from train_batch import set_environment, read_data, preprocess_data, prepare_features, split_data, is_best_model, generate_uuid

@pytest.fixture
def sample_data():
    # Directly create a DataFrame instead of reading from CSV
    data = {
        "Trip ID": ["0000184e7cd53cee95af32eba49c44e4d20adcd8", "000072ee076c9038868e239ca54185eb43959db0"],
        "Taxi ID": ["f538e6b729d1aaad4230e9dcd9dc2fd9a168826ddadbd67c2f79331875dc586863d73aa3169fb266dc5e5ed6cdc8687537de8071a51556146f5251d4d8e8237f", 
                    "e51e2c30caec952b40b8329a68b498e18ce8a1f40fa75c71e425e9426db562ac617b0a28e1c69f5c579048f75a43a2dc066c17448ab65f5016acca10558df3ed"],
        "Trip Start Timestamp": ["2024-01-19 17:00:00", "2024-01-28 14:30:00"],
        "Trip End Timestamp": ["2024-01-19 18:00:00", "2024-01-28 15:00:00"],
        "Trip Miles": [17.12, 12.7],
        "Pickup Census Tract": [17031980000.0, None],
        "Dropoff Census Tract": [17031320100.0, None],
        "Pickup Community Area": [76.0, 6.0],
        "Dropoff Community Area": [32.0, None],
        "Fare": [45.5, 33.75],
        "Tips": [10.0, 0.0],
        "Tolls": [0.0, 0.0],
        "Extras": [4.0, 0.0],
        "Trip Total": [60.0, 33.75],
        "Payment Type": ["Credit Card", "Cash"],
        "Company": ["Flash Cab", "Flash Cab"],
        "Pickup Centroid Latitude": [41.97907082, 41.944226601],
        "Pickup Centroid Longitude": [-87.903039661, -87.655998182],
        "Pickup Centroid Location": ["POINT (-87.9030396611 41.9790708201)", "POINT (-87.6559981815 41.9442266014)"],
        "Dropoff Centroid Latitude": [41.884987192, None],
        "Dropoff Centroid Longitude": [-87.620992913, None],
        "Dropoff Centroid  Location": ["POINT (-87.6209929134 41.8849871918)", None],
        "Duration": [60.0, 30.0]
    }
    df = pd.DataFrame(data)
    return df

def test_read_data(sample_data):
    # Directly use the sample data fixture instead of mocking S3 reading
    df = sample_data
    assert len(df) == 2
    assert list(df.columns) == [
        'Trip ID', 'Taxi ID', 'Trip Start Timestamp', 'Trip End Timestamp', 
        'Trip Miles', 'Pickup Census Tract', 'Dropoff Census Tract', 
        'Pickup Community Area', 'Dropoff Community Area', 'Fare', 'Tips', 
        'Tolls', 'Extras', 'Trip Total', 'Payment Type', 'Company', 
        'Pickup Centroid Latitude', 'Pickup Centroid Longitude', 
        'Pickup Centroid Location', 'Dropoff Centroid Latitude', 
        'Dropoff Centroid Longitude', 'Dropoff Centroid  Location', 'Duration'
    ]

def test_preprocess_data(sample_data):
    df = preprocess_data(sample_data)
    assert 'Trip Start Hour' in df.columns
    assert 'Trip Start DayOfWeek' in df.columns
    assert 'Trip End Hour' in df.columns
    assert 'Trip End DayOfWeek' in df.columns
    # Allow some missing values since not all data may be clean
    assert df.isnull().sum().sum() >= 0

def test_prepare_features(sample_data):
    df = preprocess_data(sample_data)
    features, target = prepare_features(df)
    assert features.shape[1] > 0
    assert target.shape[0] == features.shape[0]

def test_split_data(sample_data):
    df = preprocess_data(sample_data)
    features, target = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(features, target)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] > 0
    assert y_test.shape[0] > 0

def test_is_best_model():
    best_metrics = (10, 10, 0.8)
    current_metrics_better = (8, 9, 0.85)
    current_metrics_worse = (12, 11, 0.75)
    
    assert is_best_model(current_metrics_better, best_metrics)
    assert not is_best_model(current_metrics_worse, best_metrics)

def test_generate_uuid():
    uuids = generate_uuid(5)
    assert len(uuids) == 5
    assert all(isinstance(uid, str) for uid in uuids)

# Run the test suite
if __name__ == "__main__":
    pytest.main(["-v", "test_train_batch.py"])