import os
import numpy as np
import imageio
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# ฟังก์ชันตรวจสอบภาพ
def load_image_with_imageio(img_path):
    try:
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

# สร้าง Custom Image Data Generator
class CustomImageDataGenerator(Sequence):
    def __init__(self, directory, batch_size=32, target_size=(224, 224), class_mode='categorical'):
        self.directory = directory
        self.batch_size = batch_size
        self.target_size = target_size
        self.class_mode = class_mode
        self.image_paths = []
        self.classes = []
        
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.lower().endswith(('jpg', 'jpeg', 'png')):
                        self.image_paths.append(os.path.join(subdir_path, filename))
                        self.classes.append(subdir)
        
        self.classes = list(set(self.classes))
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = []
        batch_labels = []

        for img_path in batch_image_paths:
            image = load_image_with_imageio(img_path)
            if image is not None:
                image = np.array(image)
                image = np.resize(image, (self.target_size[0], self.target_size[1], 3))
                batch_images.append(image)
                class_name = os.path.basename(os.path.dirname(img_path))
                batch_labels.append(self.class_indices[class_name])
        
        batch_images = np.array(batch_images) / 255.0
        batch_labels = np.array(batch_labels)
        if self.class_mode == 'categorical':
            batch_labels = np.eye(len(self.classes))[batch_labels]

        return batch_images, batch_labels

    def on_epoch_end(self):
        # การสุ่มข้อมูลหรือการจัดเรียงข้อมูลใหม่สามารถทำได้ที่นี่
        pass

# เตรียม Data Generators
train_generator = CustomImageDataGenerator(
    directory='train_data',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical'
)

validation_generator = CustomImageDataGenerator(
    directory='validation_data',
    batch_size=32,
    target_size=(224, 224),
    class_mode='categorical'
)

# ฝึกโมเดล
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[early_stopping]  # เพิ่ม EarlyStopping ที่นี่
)
