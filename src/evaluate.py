import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

def evaluate_model(model_path, data_dir):
    model = load_model(model_path)
    
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )
    
    # Evaluate the model
    y_pred = model.predict(generator)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = generator.classes
    class_labels = list(generator.class_indices.keys())
    
    # Print classification report and confusion matrix
    print("Classification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred_classes))

if __name__ == "__main__":
    model_path = 'model.h5'
    data_dir = 'validation_data'
    evaluate_model(model_path, data_dir)
