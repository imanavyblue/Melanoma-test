from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.keras

# กำหนดขนาดของ input image
input_shape = (224, 224, 3)

# สร้างโมเดล
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

# กำหนด ImageDataGenerator สำหรับการโหลดข้อมูล
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/train',  # ไดเรกทอรีข้อมูลการฝึกอบรม
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'data/val',  # ไดเรกทอรีข้อมูลการตรวจสอบ
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# กำหนด EarlyStopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # ติดตามค่า val_loss
    patience=3,  # รอการปรับปรุงเป็นเวลา 3 epoch
    restore_best_weights=True  # ใช้เวทที่ดีที่สุดที่ได้จากการฝึก
)

# เริ่มการบันทึกด้วย MLflow
with mlflow.start_run():
    # บันทึกโมเดล
    mlflow.keras.log_model(model, "model")
    
    # บันทึกพารามิเตอร์
    mlflow.log_param("learning_rate", 0.0001)
    
    # ฝึกโมเดล
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stopping]  # เพิ่ม EarlyStopping ที่นี่
    )
    
    # บันทึกผลการฝึกอบรม
    for key, values in history.history.items():
        for epoch, value in enumerate(values):
            mlflow.log_metric(f"{key}_epoch_{epoch+1}", value)

    # บันทึกโมเดล
    mlflow.keras.log_model(model, "model")
