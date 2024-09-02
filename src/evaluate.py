import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import mlflow
import mlflow.keras
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# กำหนดไดเรกทอรีข้อมูลการตรวจสอบ
val_dir = 'data/val'

# กำหนด ImageDataGenerator สำหรับการโหลดข้อมูล
val_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = val_datagen.flow_from_directory(
    val_dir,  # ไดเรกทอรีข้อมูลการตรวจสอบ
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# โหลดโมเดลจาก MLflow
model_uri = "runs:/<RUN_ID>/model"  # ใส่ RUN_ID ที่ได้จากการฝึกอบรม
model = mlflow.keras.load_model(model_uri)

# ประเมินผลลัพธ์
with mlflow.start_run():
    # ประเมินโมเดล
    eval_results = model.evaluate(validation_generator)
    
    # บันทึกเมตริก
    mlflow.log_metric("val_loss", eval_results[0])
    mlflow.log_metric("val_accuracy", eval_results[1])
    
    # คาดการณ์ผลลัพธ์
    predictions = model.predict(validation_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = validation_generator.classes
    
    # สร้าง classification report
    report = classification_report(y_true, y_pred, target_names=validation_generator.class_indices.keys(), output_dict=True)
    mlflow.log_dict(report, "classification_report.json")
    
    # สร้าง confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=validation_generator.class_indices.keys(), columns=validation_generator.class_indices.keys())
    
    # บันทึก confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    
    # บันทึกผลการประเมิน
    mlflow.log_artifact('classification_report.json')

print("Evaluation completed and results are logged.")
