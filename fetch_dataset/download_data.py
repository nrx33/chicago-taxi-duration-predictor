import os
import shutil
import boto3
import gdown
from botocore.exceptions import BotoCoreError, ClientError

# URLs and filenames for the files on Google Drive
files = [
    ("https://drive.google.com/uc?id=1S21PmfHM8M3KMTcy_4ycbZdVhuoWnzhL", "Taxi_Trips_2024_01.csv"),
    ("https://drive.google.com/uc?id=1ocXMTEfXVLoUTKIYmkPFiwaO28hgr8GG", "Taxi_Trips_2024_02.csv"),
    ("https://drive.google.com/uc?id=1XPdy9r_IcbkUAdBfXckiGzdd3vFu4SD6", "Taxi_Trips_2024_03.csv"),
    ("https://drive.google.com/uc?id=1fDjjWbdGyp2f7z33cwB32-ImzAaW_zfe", "Taxi_Trips_2024_04.csv"),
    ("https://drive.google.com/uc?id=18zCyEBA1GjMxyJjtCgJF_zYmzaLbibES", "Taxi_Trips_2024_05.csv"),
    ("https://drive.google.com/uc?id=12xKQPLdTjSP61YW1UlQN3OBmwBkPnxRc", "Taxi_Trips_2024_06.csv")
]

# Set up boto3 to use LocalStack
s3 = boto3.client(
    's3',
    region_name='us-east-1',
    endpoint_url='http://localhost:4566',  # LocalStack's default endpoint
    aws_access_key_id='test',
    aws_secret_access_key='test'
)

bucket_name = 'mlflow-bucket'
folder_name = 'dataset/'

# Function to check S3 connectivity
def check_s3_connection():
    try:
        # Attempt to list buckets to verify connection
        s3.list_buckets()
        return True
    except (BotoCoreError, ClientError) as e:
        print(f"Error connecting to S3: {e}")
        return False

# Verify S3 connection
if check_s3_connection():
    # Check if bucket exists, and create it if not
    buckets = s3.list_buckets()
    if not any(bucket['Name'] == bucket_name for bucket in buckets['Buckets']):
        s3.create_bucket(Bucket=bucket_name)

    # Download directory
    download_dir = "downloads"
    os.makedirs(download_dir, exist_ok=True)

    # Download files
    for url, filename in files:
        output_path = os.path.join(download_dir, filename)
        gdown.download(url, output=output_path, quiet=False)

    # Upload files to S3 bucket
    for filename in os.listdir(download_dir):
        file_path = os.path.join(download_dir, filename)
        s3.upload_file(file_path, bucket_name, folder_name + filename)

    # Clean up: Delete downloaded files and the directory
    shutil.rmtree(download_dir)

    print("Files have been uploaded to the S3 bucket in LocalStack and deleted locally.")
else:
    print("S3 is not accessible. Files will not be downloaded.")
