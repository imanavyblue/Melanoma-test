import os
import mlflow
import mlflow.keras
from src.train import train_model

def main():
    train_dir = 'data/train'
    val_dir = 'data/val'

    os.makedirs('mlruns', exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("MelanomaDetection")

    train_model(train_dir, val_dir)

if __name__ == "__main__":
    main()
