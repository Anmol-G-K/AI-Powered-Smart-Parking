import cv2
import pickle
import numpy as np
from tensorflow.keras.models import load_model

# Load trained CNN model
parking_model = load_model("model_final.h5")

# Class labels from model output
LABELS = {0: "empty", 1: "occupied"}

# Load video
cap = cv2.VideoCapture("car_test.mp4")

# Load pre-defined parking positions
with open("carposition.pkl", "rb") as f:
    PARKING_SPOTS = pickle.load(f)

# Parking box dimensions
SPOT_W, SPOT_H = 130, 65


def analyze_parking(frame):
    """
    Detects cars in pre-marked parking spots,
    draws results on the frame, and counts free spaces.
    """
    crops = []
    free_count = 0

    # Prepare crops for all marked spots
    for (x, y) in PARKING_SPOTS:
        roi = frame[y:y + SPOT_H, x:x + SPOT_W]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_norm = roi_resized.astype("float32") / 255.0
        crops.append(roi_norm)

    # Predict occupancy for all spots
    crops = np.array(crops)
    predictions = parking_model.predict(crops)

    # Annotate frame with results
    for i, (x, y) in enumerate(PARKING_SPOTS):
        pred_id = np.argmax(predictions[i])
        label = LABELS[pred_id]

        if label == "empty":
            box_color, thickness, txt_color = (0, 255, 0), 3, (0, 0, 0)
            free_count += 1
        else:
            box_color, thickness, txt_color = (0, 0, 255), 2, (255, 255, 255)

        # Draw parking spot rectangle
        cv2.rectangle(frame, (x, y), (x + SPOT_W, y + SPOT_H), box_color, thickness)

        # Add label background
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        tx, ty = x, y + SPOT_H - 5
        cv2.rectangle(frame, (tx, ty - text_size[1] - 4), (tx + text_size[0] + 6, ty + 2), box_color, -1)

        # Add label text
        cv2.putText(frame, label, (tx + 3, ty - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    # Display summary count
    cv2.putText(frame, f"Available: {free_count} / {len(PARKING_SPOTS)}",
                (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return frame


# Main loop
while True:
    # Restart video when it ends
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    frame = analyze_parking(frame)

    cv2.imshow("Parking Lot", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()