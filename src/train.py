import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import VGG16
from sklearn.model_selection import train_test_split
import cv2

# ฟังก์ชันสำหรับตรวจสอบและโหลดภาพด้วย OpenCV
def load_image_cv(img_path):
    image = cv2.imread(img_path)
    if image is None:
        print(f"Error loading image with OpenCV: {img_path}")
    return image

# ฟังก์ชันสำหรับตรวจสอบและโหลดภาพด้วย PIL
def load_image_pil(img_path):
    try:
        image = Image.open(img_path)
        image.verify()  # ตรวจสอบว่าภาพเป็นไฟล์ที่ถูกต้อง
        return Image.open(img_path)  # เปิดภาพใหม่
    except Exception as e:
        print(f"Error loading image with PIL: {img_path}: {e}")
        return None

# กำหนดที่อยู่ของข้อมูล
train_data_dir = 'train_data'
validation_data_dir = 'validation_data'

# การสร้าง ImageDataGenerator สำหรับการโหลดและการแปลงภาพ
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # ใช้การแบ่งข้อมูล
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# การสร้างโมเดล
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(2, activation='softmax')  # สมมุติว่ามี 2 คลาส
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# การฝึกสอนโมเดล
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

print("Training complete")
