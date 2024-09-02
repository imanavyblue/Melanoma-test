# Melanoma Detection MLOps

This repository contains the necessary files to train, deploy, and monitor a deep learning model for melanoma detection.

## Directory Structure
- `data/`: Contains training, validation, and testing datasets.
- `notebooks/`: Jupyter notebooks for initial experiments.
- `src/`: Source code for data preprocessing, model building, and training.
- `mlruns/`: MLflow experiment tracking data.
- `Dockerfile`: Docker setup for building the environment.
- `docker-compose.yml`: Docker Compose setup for running services like MLflow, Prometheus, and Grafana.
- `requirements.txt`: Dependencies required for the project.
- `run_mlflow.py`: Script to train and track the model with MLflow.
- `prometheus.yml`: Configuration for Prometheus monitoring.

## Getting Started

1. Clone the repository.
2. Run `docker-compose up` to start all services.
3. Use `run_mlflow.py` to train the model and track experiments.

## Monitoring
Prometheus and Grafana are used for monitoring the model's performance.

## Deployment
The model is deployed using a Gradio interface (not included in this example) and monitored using Prometheus and Grafana.
