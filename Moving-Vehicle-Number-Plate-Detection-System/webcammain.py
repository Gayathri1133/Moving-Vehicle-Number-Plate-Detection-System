import cv2
import numpy as np
import pytesseract
import easyocr

#numberPlateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
plat_detector =  cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
video = cv2.VideoCapture(0)

if(video.isOpened()==False):
    print('Error Reading Video')
def get_vehicle_type(plate_img):
    # Resize for better color analysis
    plate_img = cv2.resize(plate_img, (100, 30))
    avg_color = cv2.mean(plate_img)[:3]
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv.reshape(-1, 3), axis=0)
    h, s, v = avg_hsv
    # Green plate (electric)
    if 35 < h < 85 and s > 50 and v > 50:
        return "public Vehicle (Yellow Plate)"
    # Yellow plate (taxi/public)
    elif 20 < h < 35 and s > 50 and v > 50:
        return "Public/Taxi Vehicle (Yellow Plate)"
    # White plate (private)
    elif s < 50 and v > 150:
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



while True:
    ret, frame = video.read()
    if not ret or frame is None:
        print("[ERROR] Cannot read frame from webcam.")
        break

    gray_video = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plate = plat_detector.detectMultiScale(gray_video,scaleFactor=1.2,minNeighbors=5,minSize=(25,25))

    for i, (x,y,w,h) in enumerate(plate):
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),2)
        # frame[y:y+h,x:x+w] = cv2.blur(frame[y:y+h,x:x+w],ksize=(10,10))
        pad = 5
        x1, y1 = max(x - pad, 0), max(y - pad, 0)
        x2, y2 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
        plate_img = frame[y1:y2, x1:x2]
        # Preprocess for OCR
        plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        plate_gray = cv2.resize(plate_gray, (400, 120))  # Upscale for better OCR
        plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
        # Sharpen image
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        plate_sharp = cv2.filter2D(plate_gray, -1, kernel)
        # Increase contrast and brightness
        alpha = 1.5  # Contrast control
        beta = 20    # Brightness control
        plate_enhanced = cv2.convertScaleAbs(plate_sharp, alpha=alpha, beta=beta)
        # Adaptive thresholding
        plate_thresh = cv2.adaptiveThreshold(
            plate_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
        )
        # Show preprocessed plate for debugging
        cv2.imshow('Plate Preprocessed', plate_thresh)
        # OCR with Tesseract
        plate_text_tess = pytesseract.image_to_string(
            plate_thresh,
            config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        plate_text_tess = plate_text_tess.strip().replace(" ", "").replace("\n", "")
        print(f"[OCR] Extracted text: '{plate_text_tess}'")

        # OCR with EasyOCR
        reader = easyocr.Reader(['en'])
        result = reader.readtext(plate_img)
        if result:
            plate_text_easy = result[0][1].upper().replace(" ", "")
            print(f"[EasyOCR] Extracted text: '{plate_text_easy}'")
            plate_text = plate_text_easy
        else:
            plate_text = plate_text_tess

        state = get_state(plate_text)
        vehicle_type = get_vehicle_type(plate_img)
        info_text = f"{plate_text} | {state}, {vehicle_type}"

        # Display info_text on the frame
        cv2.putText(
            frame,
            info_text,
            org=(x, y + h + 20),
            fontFace=cv2.FONT_HERSHEY_COMPLEX,
            fontScale=0.7,
            color=(0, 255, 0),
            thickness=2
        )
        File = f'pic{i}.jpg'
        cv2.imwrite(File, frame[y1:y2, x1:x2])
         
    if ret == True:
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()