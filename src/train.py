import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from PIL import Image

# พาธไปยังโฟลเดอร์ที่เก็บภาพ
train_data_dir = 'train_data'
validation_data_dir = 'validation_data'

# ตรวจสอบรูปภาพก่อนนำเข้ามาใน dataset
def validate_images(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            img_path = os.path.join(subdir, file)
            try:
                img = Image.open(img_path)
                img.verify()  # ตรวจสอบความถูกต้องของรูปภาพ
            except (IOError, SyntaxError, UnidentifiedImageError) as e:
                print(f"Bad file: {img_path}, Error: {e}")
                os.remove(img_path)  # ลบไฟล์ที่เสียหาย

# ตรวจสอบรูปภาพใน training และ validation set
validate_images(train_data_dir)
validate_images(validation_data_dir)

# สร้าง ImageDataGenerator สำหรับ train และ validation sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

# ใช้ InceptionV3 โมเดลในการสร้าง base model
base_model = InceptionV3(weights='imagenet', include_top=False)

# เพิ่ม layer ต่าง ๆ ตามที่ต้องการ
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# กำหนด optimizer และ compile โมเดล
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# บันทึกโมเดล
model.save('inceptionv3_model.h5')
