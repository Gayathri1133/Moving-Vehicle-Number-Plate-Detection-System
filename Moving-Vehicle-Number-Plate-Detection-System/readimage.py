import cv2
import numpy as np
import os
import easyocr
import pytesseract

# Load Haar Cascade
plat_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

# Define image path
img_path = 'car.png'

# Read image
img = cv2.imread(img_path)

if img is None:
    print(f"[ERROR] Could not load image from: {img_path}")
    exit()
def get_vehicle_type(plate_img):
    plate_img = cv2.resize(plate_img, (100, 30))
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    h, s, v = avg_hsv
    print(f"[DEBUG] HSV: h={h}, s={s}, v={v}")  # Add this line
    # Green plate (electric)
    if 35 < h < 85 and s > 50 and v > 50:
        return "Electric Vehicle (Green Plate)"
    # Yellow plate (taxi/public)
    elif 20 < h < 45 and s > 50 and v > 50:
        return "Public/Taxi Vehicle (Yellow Plate)"
    # White plate (private)
    elif h>85 and s < 50 and v < 150:
        return "Private/Own Vehicle (White Plate)"
    else:
        return "Unknown Vehicle Type"
def get_state(plate_text):
    state_codes = {
        "AP": "Andhra Pradesh",
        "TS": "Telangana",
        "IN": "Tamil Nadu",
        "KL": "Kerala",
        "DL": "Delhi",



        # Add more state codes as needed
    }
    plate_text = plate_text.upper().strip()
    plate_text = plate_text.replace(" ", "") 
    for code in state_codes:
        if plate_text.startswith(code):
            return state_codes[code]
    return "Unknown State"


def correct_plate_text(text):
    corrections = {
        "AO": "D",
        "A0": "D",
        "0": "Q",
        "1": "I",
        "O": "Q",  # Sometimes O is Q
        # Add more as needed
    }
    # Replace longer patterns first
    for wrong, right in sorted(corrections.items(), key=lambda x: -len(x[0])):
        text = text.replace(wrong, right)
    return text

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect plates
plates = plat_detector.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=5, minSize=(25, 25)
)

# Draw bounding boxes and save cropped plates
for i, (x, y, w, h) in enumerate(plates):
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    plate_img = img[y:y + h, x:x + w]
    # Preprocess for OCR
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.resize(plate_gray, (400, 120))
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    plate_sharp = cv2.filter2D(plate_gray, -1, kernel)
    alpha = 1.5
    beta = 20
    plate_enhanced = cv2.convertScaleAbs(plate_sharp, alpha=alpha, beta=beta)
    plate_thresh = cv2.adaptiveThreshold(
        plate_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    # OCR with Tesseract only
    plate_text = pytesseract.image_to_string(
        plate_thresh,
        config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    )
    plate_text = plate_text.strip().replace(" ", "").replace("\n", "")
    plate_text = correct_plate_text(plate_text)
    print(f"[OCR] Extracted text: '{plate_text}'")
    state = get_state(plate_text)
    vehicle_type = get_vehicle_type(plate_img)
    info_text = f"{plate_text} | {state}, {vehicle_type}"
    cv2.putText(
        img,
        info_text,
        org=(x, y + h + 20),
        fontFace=cv2.FONT_HERSHEY_COMPLEX,
        fontScale=0.7,
        color=(0, 255, 0),
        thickness=2
    )
    print(f"Saving cropped plate: plate_{i}.jpg, shape: {plate_img.shape}")
    success = cv2.imwrite(f'plate_{i}.jpg', plate_img)
    if not success:
        print(f"[ERROR] Failed to save plate_{i}.jpg")

cv2.imshow('plates', img)
cv2.waitKey(10000)
cv2.destroyAllWindows()
