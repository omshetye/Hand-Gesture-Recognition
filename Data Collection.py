import cv2  # Install opencv-python
import numpy as np
import mediapipe as mp
counter = 0


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

camera = cv2.VideoCapture(2)


while True:
    # Grab the webcam's image.
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    white_img = np.zeros_like(frame)
    white_img.fill(255)
    drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(white_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=drawing_spec)
    cv2.imshow("IMG2", frame)
    cv2.imshow("IMG", white_img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        counter += 1
        print(counter)
        cv2.imwrite(f'data_om/Z/{counter}.jpg', white_img)
