import cv2
import pickle
import os

# Parking spot dimensions
SPOT_W, SPOT_H = 130, 65

# Directory to save cropped ROIs
OUTPUT_DIR = "cropped_img"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_roi(image, position, idx):
    """Save a cropped parking spot image for dataset building."""
    x, y = position
    roi = image[y:y + SPOT_H, x:x + SPOT_W]
    save_path = os.path.join(OUTPUT_DIR, f"spot_{idx}.png")
    cv2.imwrite(save_path, roi)
    print(f"[INFO] ROI saved at: {save_path}")


# Load saved positions if available
try:
    with open("carposition.pkl", "rb") as f:
        PARKING_POSITIONS = pickle.load(f)
except FileNotFoundError:
    PARKING_POSITIONS = []


def mouse_event(event, x, y, flags, param):
    """
    Mouse callback:
    - Left click: add parking spot
    - Right click: remove spot if clicked inside
    """
    global PARKING_POSITIONS

    if event == cv2.EVENT_LBUTTONDOWN:
        PARKING_POSITIONS.append((x, y))
        test_img = cv2.resize(cv2.imread("car1.png"), (1280, 720))
        save_roi(test_img, (x, y), len(PARKING_POSITIONS))

    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, (x1, y1) in enumerate(PARKING_POSITIONS):
            if x1 < x < x1 + SPOT_W and y1 < y < y1 + SPOT_H:
                PARKING_POSITIONS.pop(i)
                break

    with open("carposition.pkl", "wb") as f:
        pickle.dump(PARKING_POSITIONS, f)


# Main loop
while True:
    frame = cv2.imread("car1.png")
    frame = cv2.resize(frame, (1280, 720))

    # Draw rectangles on all marked spots
    for (x, y) in PARKING_POSITIONS:
        cv2.rectangle(frame, (x, y), (x + SPOT_W, y + SPOT_H), (255, 0, 255), 2)

    cv2.imshow("Mark Parking Spots", frame)
    cv2.setMouseCallback("Mark Parking Spots", mouse_event)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
