
import threading
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np

from django.http import HttpResponse
from django.views.decorators.http import require_POST


# ==========================================
# FILE PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ML_DATA" / "asl_model.joblib"


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# MEDIAPIPE HAND DETECTION
# ==========================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# Prevent overlapping requests from using Hands simultaneously.
hands_lock = threading.Lock()


# ==========================================
# PREPROCESS HAND LANDMARKS
# ==========================================

def preprocess_landmarks(hand_landmarks):
    landmarks = np.array(
        [
            [point.x, point.y]
            for point in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    # Make coordinates relative to the wrist.
    landmarks -= landmarks[0].copy()

    # Normalize coordinates.
    max_value = np.max(np.abs(landmarks))

    if max_value > 0:
        landmarks /= max_value

    return landmarks.flatten().reshape(1, -1)


# ==========================================
# ASL PREDICTION VIEW
# ==========================================

@require_POST
def predict(request):

    uploaded_file = request.FILES.get("frame")

    if uploaded_file is None:
        return HttpResponse(
            "No frame received",
            status=400
        )

    # Decode uploaded image.
    image_bytes = np.frombuffer(
        uploaded_file.read(),
        dtype=np.uint8
    )

    frame = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return HttpResponse(
            "Invalid image",
            status=400
        )

    # Mirror the image.
    frame = cv2.flip(frame, 1)

    # Default prediction.
    letter = "-"
    confidence = 0.0
    status = "Show your hand"

    
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb_frame.flags.writeable = False

    
    with hands_lock:
        results = hands.process(rgb_frame)

    rgb_frame.flags.writeable = True

    

    if results.multi_hand_landmarks:

        
        hand_landmarks = results.multi_hand_landmarks[0]

        
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # Extract the 42 hand features.
        features = preprocess_landmarks(
            hand_landmarks
        )

        try:
            # Predict ASL letter.
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(
                    features
                )[0]

                best_index = int(
                    np.argmax(probabilities)
                )

                confidence = float(
                    probabilities[best_index]
                )

                prediction = model.classes_[best_index]

            else:
                prediction = model.predict(
                    features
                )[0]

                confidence = 1.0

            # Confidence threshold.
            if confidence >= 0.60:
                letter = str(prediction)
                status = "Hand detected"

            else:
                letter = "?"
                status = "Low confidence"

        except (
            ValueError,
            IndexError,
            AttributeError
        ) as error:

            print(
                f"ASL prediction error: {error}"
            )

            letter = "?"
            confidence = 0.0
            status = "Prediction error"


    
    cv2.rectangle(
        frame,
        (5, 5),
        (180, 60),
        (20, 25, 35),
        -1
    )

    
    cv2.putText(
        frame,
        f"Letter: {letter}",
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 100),
        2
    )

    
    cv2.putText(
        frame,
        f"Confidence: {confidence:.0%}",
        (15, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    

    success, encoded_image = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            55
        ]
    )

    if not success:
        return HttpResponse(
            "Could not encode image",
            status=500
        )

    response = HttpResponse(
        encoded_image.tobytes(),
        content_type="image/jpeg"
    )

    
    response["X-Letter"] = letter
    response["X-Confidence"] = str(confidence)
    response["X-Status"] = status
    response["Cache-Control"] = "no-store"

    return response
