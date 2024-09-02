import numpy as np
import os
from PIL import Image
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# ฟังก์ชันสำหรับโหลดภาพและแปลงเป็น NumPy array
def load_image(img_path, target_size=(299, 299)):
    try:
        # ใช้ PIL เพื่อโหลดภาพ
        with Image.open(img_path) as img:
            img = img.resize(target_size)
            img_array = np.array(img)
            return img_array
    except Exception as e:
        print(f"Error loading image: {img_path}, {e}")
        return None

# สร้างโมเดล InceptionV3
base_model = InceptionV3(weights='imagenet', include_top=False)

# เพิ่มเลเยอร์ของเราด้านบน
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # สมมติว่ามี 2 classes
model = Model(inputs=base_model.input, outputs=predictions)

# แช่แข็งเลเยอร์ของ InceptionV3 ไม่ให้ถูกฝึกสอนใหม่
for layer in base_model.layers:
    layer.trainable = False

# คอมไพล์โมเดล
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# กำหนดที่อยู่ของข้อมูล
train_data_dir = 'train_data'
validation_data_dir = 'validation_data'

# สร้าง ImageDataGenerator สำหรับการโหลดและการแปลงภาพ
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # ใช้การแบ่งข้อมูล
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# การฝึกสอนโมเดล
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

print("Training complete")
