import os
import tensorflow as tf
import optuna
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import mlflow
import mlflow.keras

# กำหนดพารามิเตอร์ของโมเดล
input_shape = (224, 224, 3)
num_classes = 2  # จำนวนคลาสที่คุณต้องการจำแนก

# สร้างโมเดล
def create_model(learning_rate):
    input_tensor = Input(shape=input_shape)
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    for layer in base_model.layers:
        layer.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=input_tensor, outputs=output)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.optimizers.SGD(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model

# เตรียมข้อมูลสำหรับฝึกอบรมและ validation
def prepare_data():
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
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        'dataset/train',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        'dataset/val',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )
    
    return train_generator, validation_generator

# ฝึกอบรมโมเดล
def train_model(trial):
    # ใช้ Optuna เพื่อค้นหาพารามิเตอร์ที่ดีที่สุด
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    
    model = create_model(learning_rate)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    model_checkpoint = ModelCheckpoint(
        'best_model.h5',
        save_best_only=True,
        save_weights_only=False
    )
    
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )
    
    # การประเมินโมเดล
    val_loss = min(history.history['val_loss'])
    return val_loss

def objective(trial):
    # เริ่มต้น MLflow run
    mlflow.start_run()
    
    # เตรียมข้อมูล
    global train_generator, validation_generator
    train_generator, validation_generator = prepare_data()
    
    # ฝึกอบรมโมเดล
    val_loss = train_model(trial)

    # บันทึกผลลัพธ์ใน MLflow
    mlflow.log_params({'learning_rate': trial.params['learning_rate']})
    mlflow.log_metrics({'val_loss': val_loss})
    
    mlflow.end_run()
    
    return val_loss

def main():
    # การปรับพารามิเตอร์ด้วย Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)
    
    print('Best trial:')
    trial = study.best_trial
    print(f'  Value: {trial.value}')
    print(f'  Params: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')

if __name__ == "__main__":
    main()
