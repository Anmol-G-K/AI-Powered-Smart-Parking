from picamera2 import Picamera2
import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify
import pickle
from keras.models import load_model
app = Flask(__name__)

model = load_model('model_final.h5')

class_dictionary = {0: 'no_car', 1: 'car'}

cap = cv2.VideoCapture('car_test.mp4')

with open('carposition.pkl', 'rb') as f:
    posList = pickle.load(f)

width, height = 130, 65

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()

def checkParkingSpace(img_bgr):
    spaceCounter = 0
    imgCrops = []
    for x, y in posList:
        crop = img_bgr[y:y+height, x:x+width]
        crop = cv2.resize(crop, (48, 48))
        crop = crop / 255.0
        imgCrops.append(crop)
    if not imgCrops:
        return img_bgr, 0, 0
    preds = model.predict(np.array(imgCrops), verbose=0)
    for i, (x, y) in enumerate(posList):
        label = class_dictionary[int(np.argmax(preds[i]))]
        if label == 'no_car':
            color, thickness, textColor = (0,255,0), 5, (0,0,0)
            spaceCounter += 1
        else:
            color, thickness, textColor = (0,0,255), 2, (255,255,255)
        cv2.rectangle(img_bgr, (x, y), (x+width, y+height), color, thickness)
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx, ty = x, y + height - 5
        cv2.rectangle(img_bgr, (tx, ty - ts[1] - 5), (tx + ts[0] + 6, ty + 2), color, -1)
        cv2.putText(img_bgr, label, (tx + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)
    total = len(posList)
    return img_bgr, spaceCounter, total - spaceCounter

def generate_frames():
    while True:
        frame_rgb = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        img, _, _ = checkParkingSpace(frame_bgr)
        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/space_count')
def space_count():
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    _, free_spaces, occupied_spaces = checkParkingSpace(frame_bgr)
    return jsonify(free=free_spaces, occupied=occupied_spaces)

if __name__ == "__main__":
    import os
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=5000, debug=debug)