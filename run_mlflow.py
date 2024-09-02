import os
import mlflow
import mlflow.keras
from src.train import train_model

def main():
    train_dir = 'train_data'
    val_dir = 'validation_data'

    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MelanomaDetection")

    train_model(train_dir, val_dir)

if __name__ == "__main__":
    main()
