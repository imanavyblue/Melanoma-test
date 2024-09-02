import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ฟังก์ชันตรวจสอบภาพ
def load_image_with_imageio(img_path):
    try:
        import imageio
        image = imageio.imread(img_path)
        return image
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        return None

# สร้างโมเดล
input_shape = (224, 224, 3)

input_tensor = Input(shape=input_shape)
base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
for layer in base_model.layers:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
output = Dense(2, activation='softmax')(x)
model = Model(inputs=input_tensor, outputs=output)
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.optimizers.SGD(learning_rate=0.0001),
    metrics=['accuracy']
)

model.summary()

# กำหนด EarlyStopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # ติดตามค่า val_loss
    patience=3,  # รอการปรับปรุงเป็นเวลา 3 epoch
    restore_best_weights=True  # ใช้เวทที่ดีที่สุดที่ได้จากการฝึก
)

# เตรียม Data Generators
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

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    'validation_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ฝึกโมเดล
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]  # เพิ่ม EarlyStopping ที่นี่
)
