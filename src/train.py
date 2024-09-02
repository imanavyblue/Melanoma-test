from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from model import create_model
import tensorflow as tf
import os
from PIL import Image

def is_image_valid(image_path):
    try:
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img.verify()  # ตรวจสอบว่าเป็นภาพที่ถูกต้อง
    except (IOError, SyntaxError) as e:
        print(f'ไม่สามารถโหลดภาพ {image_path}: {e}')
        return False
    return True

def create_valid_image_generator(directory, target_size, batch_size):
    valid_image_paths = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                file_path = os.path.join(subdir, file)
                if is_image_valid(file_path):
                    valid_image_paths.append(file_path)
    
    # ใช้ ImageDataGenerator โดยให้เฉพาะไฟล์ภาพที่ถูกต้อง
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        dataframe=pd.DataFrame(valid_image_paths, columns=['filename']),
        directory=None,
        x_col='filename',
        y_col=None,
        target_size=target_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False
    )
    return generator

def train_model(train_dir, val_dir, epochs=10):
    # Data augmentation and preparation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_generator = create_valid_image_generator(
        train_dir,
        target_size=(224, 224),
        batch_size=32
    )

    validation_generator = create_valid_image_generator(
        val_dir,
        target_size=(224, 224),
        batch_size=32
    )

    model = create_model()

    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping]
    )
    model.save("Inception_V3.h5")

if __name__ == "__main__":
    train_dir = 'train_data'
    val_dir = 'validation_data'
    train_model(train_dir, val_dir)
