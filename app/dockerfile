# Use an official Python runtime as a parent image
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY train_batch.py /app/
COPY requirements.txt /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libyaml-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --upgrade pip setuptools wheel

# Install Cython separately (often needed for PyYAML)
RUN pip install Cython

# Install Python dependencies
RUN pip install -r requirements.txt

# Remove development dependencies to reduce image size
RUN apt-get remove -y build-essential libssl-dev libffi-dev python3-dev libyaml-dev git && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Make port 80 available to the world outside this container
EXPOSE 80

# Set the entrypoint to python
ENTRYPOINT ["python"]

# Set the default command to run train_batch.py
CMD ["train_batch.py"]