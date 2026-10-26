
import cv2
import joblib
import mediapipe as mp
import numpy as np

from pathlib import Path
from django.http import HttpResponse
from django.views.decorators.http import require_POST



from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "ML_DATA" / "asl_model.joblib"

model = joblib.load(MODEL_PATH)




mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)




def preprocess_landmarks(hand_landmarks):
    landmarks = np.array(
        [
            [landmark.x, landmark.y]
            for landmark in hand_landmarks.landmark
        ],
        dtype=np.float32
    )

    # Wrist-relative coordinates
    landmarks[:, 0] -= landmarks[0, 0]
    landmarks[:, 1] -= landmarks[0, 1]

    # Normalize
    max_value = np.max(np.abs(landmarks))

    if max_value > 0:
        landmarks /= max_value

    return landmarks.flatten().reshape(1, -1)




@require_POST
def predict(request):
    uploaded_file = request.FILES.get("frame")

    if uploaded_file is None:
        return HttpResponse(
            "No frame received",
            status=400
        )

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

    # Mirror like a selfie camera
    frame = cv2.flip(frame, 1)

    # Detect hands
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    letter = "-"
    confidence = 0.0
    status = "Show your hand"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = preprocess_landmarks(
                hand_landmarks
            )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(
                    features
                )

                best_index = int(
                    np.argmax(probabilities[0])
                )

                confidence = float(
                    probabilities[0][best_index]
                )

                prediction = model.classes_[best_index]

            else:
                prediction = model.predict(features)[0]
                confidence = 1.0

            if confidence >= 0.30:
                letter = str(prediction)
                status = "Hand detected"
            else:
                letter = "?"
                status = "Low confidence"

    # Draw prediction information
    cv2.rectangle(
        frame,
        (5, 5),
        (250, 80),
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
        [cv2.IMWRITE_JPEG_QUALITY, 65]
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
