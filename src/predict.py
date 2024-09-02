import gradio as gr
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from prometheus_client import start_http_server, Summary, Gauge
import time

# โหลดโมเดล InceptionV3
model = tf.keras.models.load_model("Inception_V3.h5")

# สมมติว่ามีชื่อคลาสแบบกำหนดเอง
class_names = ["Benign", "Malignant"]  # ปรับชื่อคลาสตามที่คุณฝึกโมเดล

# สร้าง Metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
INFERENCE_COUNT = Gauge('inference_count', 'Number of inferences processed')

# ฟังก์ชันสำหรับการพยากรณ์
@REQUEST_TIME.time()
def predict(img):
    INFERENCE_COUNT.inc()
    img = img.resize((224, 224))  # ปรับขนาดรูปภาพ
    img_array = image.img_to_array(img)  # แปลงรูปภาพเป็นอาร์เรย์
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติแบทช์
    img_array = preprocess_input(img_array)  # เตรียมรูปภาพให้สอดคล้องกับความต้องการของโมเดล

    predictions = model.predict(img_array)
    predictions = predictions[0]  # เอาค่าผลลัพธ์ของ batch เดียว
    confidence_dict = {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

    return confidence_dict

# สร้าง Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Label(num_top_classes=2, label="Predicted Class"),
    title="Melanoma Classification with InceptionV3",
    description="Upload an image to classify it into one of the classes."
)

# เริ่มเซิร์ฟเวอร์ metrics
def start_metrics_server():
    start_http_server(8000)

if __name__ == "__main__":
    # เริ่มเซิร์ฟเวอร์ metrics
    start_metrics_server()
    
    # เริ่ม Gradio App
    interface.launch()
