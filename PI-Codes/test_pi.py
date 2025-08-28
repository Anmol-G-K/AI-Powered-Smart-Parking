import cv2
import numpy as np
import pickle
from keras.models import load_model
from picamera2 import Picamera2

model = load_model("model_final.h5")
class_dictionary = {0: 'no_car', 1: 'car'}

with open('carposition.pkl', 'rb') as f:
    positionList = pickle.load(f)

width, height = 130, 65

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()

def checkingCarParking(image_bgr):
    imgCrops = []
    spaceCounter = 0
    for x, y in positionList:
        crop = image_bgr[y:y+height, x:x+width]
        crop = cv2.resize(crop, (48, 48))
        crop = crop / 255.0
        imgCrops.append(crop)

    if not imgCrops:
        return image_bgr

    preds = model.predict(np.array(imgCrops), verbose=0)

    for i, (x, y) in enumerate(positionList):
        label = class_dictionary[int(np.argmax(preds[i]))]
        if label == 'no_car':
            color, thickness, textColor = (0,255,0), 5, (0,0,0)
            spaceCounter += 1
        else:
            color, thickness, textColor = (0,0,255), 2, (255,255,255)
        cv2.rectangle(image_bgr, (x, y), (x+width, y+height), color, thickness)
        ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx, ty = x, y + height - 5
        cv2.rectangle(image_bgr, (tx, ty - ts[1] - 5), (tx + ts[0] + 6, ty + 2), color, -1)
        cv2.putText(image_bgr, label, (tx + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, textColor, 1)

    cv2.putText(image_bgr, f'Space Count: {spaceCounter}', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    return image_bgr

while True:
    frame_rgb = picam2.capture_array()
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    image = checkingCarParking(frame_bgr)
    cv2.imshow("Image", image)
    if cv2.waitKey(10) == ord('q'):
        break

cv2.destroyAllWindows()