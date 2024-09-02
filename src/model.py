from keras.applications import InceptionV3
from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
import tensorflow as tf
from data_preprocessing import preprocess_data

def create_model(input_shape=(224, 224, 3)):
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
    return model

if __name__ == "__main__":
    model = create_model()

