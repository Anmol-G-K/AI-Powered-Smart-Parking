from flask import Flask, render_template, Response, jsonify
import cv2
import pickle
import numpy as np
from keras.models import load_model
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained classification model
parking_model = load_model("model_final.h5")

# Labels corresponding to model output
LABELS = {0: "empty", 1: "occupied"}

# Video source
video_stream = cv2.VideoCapture("car_test.mp4")

# Load pre-marked parking positions
with open("carposition.pkl", "rb") as f:
    PARKING_SPOTS = pickle.load(f)

# Dimensions of each parking region
SPOT_W, SPOT_H = 130, 65

# Load college logo
LOGO_PATH = "static/assets/college_logo.png"
college_logo = None
if os.path.exists(LOGO_PATH):
    college_logo = cv2.imread(LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if college_logo is not None:
        # Resize logo to appropriate size (adjust as needed)
        logo_height, logo_width = college_logo.shape[:2]
        target_height = 80
        target_width = int((logo_width / logo_height) * target_height)
        college_logo = cv2.resize(college_logo, (target_width, target_height))


def analyze_frame(frame):
    """
    Checks all marked parking spaces in a given frame,
    classifies them as 'empty' or 'occupied', and draws bounding boxes.
    Returns processed frame + counts of free and occupied spaces.
    """
    free_count = 0
    crops = []

    # Extract image crops for each parking position
    for (x, y) in PARKING_SPOTS:
        roi = frame[y:y + SPOT_H, x:x + SPOT_W]
        roi_resized = cv2.resize(roi, (48, 48))
        roi_norm = roi_resized.astype("float32") / 255.0
        crops.append(roi_norm)

    # Run model predictions
    crops = np.array(crops)
    predictions = parking_model.predict(crops)

    # Annotate the frame with bounding boxes + labels
    for i, (x, y) in enumerate(PARKING_SPOTS):
        pred_label = LABELS[np.argmax(predictions[i])]

        if pred_label == "empty":
            color, thickness, txt_color = (0, 255, 0), 4, (0, 0, 0)
            free_count += 1
        else:
            color, thickness, txt_color = (0, 0, 255), 2, (255, 255, 255)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + SPOT_W, y + SPOT_H), color, thickness)

        # Draw label background + text
        label_size = cv2.getTextSize(pred_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x, text_y = x, y + SPOT_H - 5
        cv2.rectangle(frame,
                      (text_x, text_y - label_size[1] - 4),
                      (text_x + label_size[0] + 6, text_y + 2),
                      color, -1)
        cv2.putText(frame, pred_label, (text_x + 3, text_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt_color, 1)

    total_spots = len(PARKING_SPOTS)
    
    # # Add college logo overlay to the left side of the frame
    # if college_logo is not None:
    #     try:
    #         # Position logo in top-left corner with some padding
    #         logo_height, logo_width = college_logo.shape[:2]
    #         x_offset, y_offset = 1000, 20
            
    #         # Create a region of interest (ROI) for the logo
    #         roi = frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width]
            
    #         # If logo has alpha channel (PNG with transparency)
    #         if college_logo.shape[2] == 4:
    #             # Extract alpha channel
    #             alpha = college_logo[:, :, 3] / 255.0
    #             alpha = np.expand_dims(alpha, axis=-1)
                
    #             # Blend logo with frame
    #             blended = (college_logo[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
    #             frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = blended
    #         else:
    #             # If no alpha channel, just overlay the logo
    #             frame[y_offset:y_offset + logo_height, x_offset:x_offset + logo_width] = college_logo[:, :, :3]
    #     except Exception as e:
    #         print(f"Error overlaying logo: {e}")
    
    return frame, free_count, total_spots - free_count


def frame_generator():
    """Stream frames to browser with bounding boxes."""
    while True:
        success, frame = video_stream.read()
        if not success:
            break

        frame = cv2.resize(frame, (1280, 720))
        frame, _, _ = analyze_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


@app.route("/")
def home():
    """Render dashboard page."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video stream route for browser."""
    return Response(frame_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/space_count")
def get_space_count():
    """Return current free/occupied space count as JSON."""
    success, frame = video_stream.read()
    if success:
        frame = cv2.resize(frame, (1280, 720))
        _, free, occupied = analyze_frame(frame)
        return jsonify(free=free, occupied=occupied)
    return jsonify(free=0, occupied=0)


if __name__ == "__main__":
    app.run(debug=True)
