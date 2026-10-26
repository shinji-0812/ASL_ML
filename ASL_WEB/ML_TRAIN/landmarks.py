
import csv
import os
import cv2
import mediapipe as mp

DATA_DIR = "dataset"
CSV_PATH = os.path.join(DATA_DIR, "asl_landmarks.csv")

os.makedirs(DATA_DIR, exist_ok=True)

LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def preprocess_landmarks(hand_landmarks):
    points = []

    for landmark in hand_landmarks.landmark:
        points.append([landmark.x, landmark.y])

    base_x, base_y = points[0]

    for point in points:
        point[0] -= base_x
        point[1] -= base_y

    max_value = max(abs(value) for point in points for value in point)

    if max_value == 0:
        return None

    return [
        coordinate / max_value
        for point in points
        for coordinate in point
    ]


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam. Check camera permissions.")
        return

    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["label"] + [f"feature_{i}" for i in range(42)]
            )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    ) as hands:

        print("SignSpeak ASL Data Collector")
        print("Show a hand sign, then press its letter key.")
        print("Press ESC to exit.")

        while True:
            success, frame = camera.read()

            if not success:
                print("Failed to read webcam frame.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            features = None

            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]

                mp_drawing.draw_landmarks(
                    frame, hand, mp_hands.HAND_CONNECTIONS
                )

                if len(results.multi_hand_landmarks) >= 2:
                    hand2 = results.multi_hand_landmarks[1]
                    mp_drawing.draw_landmarks(
                        frame, hand2, mp_hands.HAND_CONNECTIONS
                    )

                features = preprocess_landmarks(hand)

            cv2.putText(
                frame,
                "Show sign, then press A-Z to save",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "ESC: Exit",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("SignSpeak Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

            if key in [ord(letter.lower()) for letter in LETTERS]:
                if features is None:
                    print("No hand detected. Sample not saved.")
                    continue

                letter = chr(key).upper()

                with open(CSV_PATH, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([letter] + features)

                print(f"Saved sample for {letter}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
