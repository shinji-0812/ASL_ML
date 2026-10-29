
import csv
import os

import cv2
import mediapipe as mp




DATA_DIR = "ASL_WEB/ML_TRAIN/dataset"
CSV_PATH = os.path.join(DATA_DIR, "asl_data.csv")

os.makedirs(DATA_DIR, exist_ok=True)




LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")




mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils




def preprocess_landmarks(hand_landmarks):
    points = [
        [landmark.x, landmark.y]
        for landmark in hand_landmarks.landmark
    ]

    base_x, base_y = points[0]

    for point in points:
        point[0] -= base_x
        point[1] -= base_y


    max_value = max(
        abs(value)
        for point in points
        for value in point
    )

    if max_value == 0:
        return None

    features = [
        coordinate / max_value
        for point in points
        for coordinate in point
    ]

    return features




def create_csv():
    if not os.path.exists(CSV_PATH):
        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                ["label"] +
                [f"feature_{i}" for i in range(42)]
            )




def main():
    create_csv()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam. Check camera permissions.")
        return

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        ) as hands:

            print("\nSignSpeak ASL Data Collector")
            print("-----------------------------")
            print("Show ONE hand sign.")
            print("Press A-Z to save a sample.")
            print("SPACE = Save space label")
            print("ENTER = Save enter label")
            print("ESC = Exit")
            print(f"\nDataset: {CSV_PATH}\n")

            while True:
                success, frame = camera.read()

                if not success:
                    print("Failed to read webcam frame.")
                    break

                # Mirror the webcam.
                frame = cv2.flip(frame, 1)

                # MediaPipe requires RGB.
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                results = hands.process(rgb)

                features = None

                # Detect and draw only hands.
                if results.multi_hand_landmarks:
                    hand = results.multi_hand_landmarks[0]

                    mp_drawing.draw_landmarks(
                        frame,
                        hand,
                        mp_hands.HAND_CONNECTIONS
                    )

                    features = preprocess_landmarks(hand)

                # Display instructions.
                cv2.putText(
                    frame,
                    "A-Z: Save ASL sample",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    "SPACE: Space | ENTER: Enter",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "ESC: Exit",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        "HAND DETECTED"
                        if features is not None
                        else "SHOW YOUR HAND"
                    ),
                    (10, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (
                        (0, 255, 0)
                        if features is not None
                        else (0, 0, 255)
                    ),
                    2
                )

                cv2.imshow(
                    "SignSpeak Data Collector",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == 27:
                    break

                label = None

                # Save A-Z samples.
                if key in [
                    ord(letter.lower())
                    for letter in LETTERS
                ]:
                    label = chr(key).upper()

                # Save SPACE label.
                elif key == 32:
                    label = "space"

                # Save ENTER label.
                elif key in [10, 13]:
                    label = "enter"

                # Save sample only when a hand is detected.
                if label is not None:
                    if features is None:
                        print(
                            "No hand detected. Sample not saved."
                        )
                        continue

                    with open(
                        CSV_PATH,
                        "a",
                        newline="",
                        encoding="utf-8"
                    ) as file:
                        writer = csv.writer(file)
                        writer.writerow([label] + features)

                    print(f"Saved sample for {label}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
