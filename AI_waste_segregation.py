import cv2
import numpy as np
import urllib.request
import serial
import time
import sys
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

CONFIDENCE_THRESHOLD = 0.55  
EXIT_TIMEOUT = 40
CAMERA_URL = "http://10.203.233.215:8080/shot.jpg"
MODEL_PATH = "mobilenetv2_waste_detection.h5"
SERIAL_PORT = "COM3"
BAUD_RATE = 9600

CLASS_NAMES = ['Recyclable', 'biodegradable', 'hazardous']
MAPPING = {'Recyclable': 'R', 'biodegradable': 'B', 'hazardous': 'H'}

try:
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
except Exception as e:
    sys.exit()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    sys.exit()

def send_with_ack(message):
    arduino.reset_input_buffer()
    arduino.reset_output_buffer()
    arduino.write((message + "\n").encode())
    arduino.flush()
    start = time.time()
    while time.time() - start < 3:  
        if arduino.in_waiting > 0:
            if arduino.readline().decode(errors='ignore').strip() == "ACK":
                return True
        time.sleep(0.05)
    return False

def capture_and_predict():
    try:
        
        img_resp = urllib.request.urlopen(CAMERA_URL, timeout=3)  
        img_np = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        img = cv2.imdecode(img_np, -1)
        if img is None:
            return None

        
        h, w = img.shape[:2]
        img = img[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)]

        # CLAHE
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
        img = cv2.cvtColor(cv2.merge((l,a,b)), cv2.COLOR_LAB2BGR)

        # Preprocess
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocess_input(img.astype(np.float32))
        img = np.expand_dims(img, axis=0)

        # Predict
        result = model.predict(img, verbose=0)
        confidence = float(np.max(result[0]))
        label = CLASS_NAMES[np.argmax(result[0])]
        print(f"{label} {confidence:.2%}")

        if confidence < CONFIDENCE_THRESHOLD:
            return None
        return MAPPING[label]

    except Exception as e:
        return None

print("Ready")
LAST_ACTIVITY = time.time()

while True:
    if time.time() - LAST_ACTIVITY > EXIT_TIMEOUT:
        arduino.close()
        sys.exit()
    try:
        if arduino.in_waiting > 0:
            signal = arduino.readline().decode('utf-8', errors='ignore').strip()
            if signal == "CAPTURE":
                LAST_ACTIVITY = time.time()
                arduino.reset_input_buffer()

                result = capture_and_predict()      
                if result is None:
                    result = capture_and_predict()  

                send_with_ack(result if result else "RETRY")

                arduino.reset_input_buffer()
                arduino.reset_output_buffer()
                LAST_ACTIVITY = time.time()

    except serial.SerialException:
        time.sleep(2)
    except Exception:
        time.sleep(1)