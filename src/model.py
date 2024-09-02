from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
import tensorflow as tf

def build_model(input_shape=(224, 224, 3), num_classes=2):
    # สร้าง input tensor
    input_tensor = Input(shape=input_shape)

    # โหลดโมเดล InceptionV3 ที่ pretrained จาก ImageNet โดยไม่รวม top layer
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=input_tensor)
    
    # ทำให้ layers ของ base model ไม่สามารถเรียนรู้ได้
    for layer in base_model.layers:
        layer.trainable = False

    # เพิ่ม Global Average Pooling layer
    x = GlobalAveragePooling2D()(base_model.output)
    
    # เพิ่ม Dense layers สำหรับ classification
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    
    # Output layer สำหรับการจำแนกประเภท 2 ประเภท
    output = Dense(num_classes, activation='softmax')(x)
    
    # สร้างโมเดล
    model = Model(inputs=input_tensor, outputs=output)
    
    # คอมไพล์โมเดล
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.optimizers.SGD(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    # แสดงสรุปของโมเดล
    model.summary()
    
    return model
