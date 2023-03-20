from keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
np.set_printoptions(suppress=True)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

model = load_model(f"Model/keras_model.h5", compile=False)
class_names = open(f"Model/labels.txt", "r").readlines()

camera = cv2.VideoCapture(2)
delay = int(1000/60)

while True:

    # Grab the webcam's image.
    ret, frame = camera.read()
    frame = cv2.flip(frame, 1)

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Resize the raw image into (224-height,224-width) pixels
    frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)

    white_img = np.zeros_like(frame)
    white_img.fill(255)
    drawing_spec = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(white_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=drawing_spec)

    ip = np.asarray(white_img, dtype=np.float32).reshape(1, 224, 224, 3)
    ip = (ip.astype(np.float32) / 127.5) - 1

    prediction = model.predict(ip)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")

    output_label = class_name[2:] + " " + str(np.round(confidence_score * 100))[:-2] + "%"
    label_text = f"{output_label}"
    label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    label_x, label_y = 10, 30
    cv2.putText(white_img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    white_img_resized = cv2.resize(white_img, (300, 300))
    cv2.imshow("Webcam Image", white_img_resized)
    cv2.imshow("frame", frame)
    if cv2.waitKey(delay) == ord('q'):
        break












    # ip = np.asarray(white_img, dtype=np.float32).reshape(1, 224, 224, 3)
    # ip = (ip / 255) - 1
    # prediction = model.predict(ip)
    # index = np.argmax(prediction)
    # class_name = class_names[index]
    # confidence_score = prediction[0][index]
    # print("Class:", class_name[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score * 100))[:-2], "%")
    #
    # output_label = class_name[2:] + " " + str(np.round(confidence_score * 100))[:-2] + "%"
    # label_text = f"{output_label}"
    # label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # label_x, label_y = 10, 30
    # cv2.putText(white_img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)





    # Predicts the model
    # prediction1 = model1.predict(ip)
    # prediction2 = model2.predict(ip)
    # index1 = np.argmax(prediction1)
    # index2 = np.argmax(prediction2)
    # class_name1 = class_names1[index1]
    # class_name2 = class_names2[index2]
    # confidence_score1 = prediction1[0][index1]
    # confidence_score2 = prediction2[0][index2]
    #
    # # Print prediction and confidence score
    # print("Class:", class_name1[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score1 * 100))[:-2], "%")
    #
    # print("Class:", class_name2[2:], end="")
    # print("Confidence Score:", str(np.round(confidence_score2 * 100))[:-2], "%")
    #
    # output_label = class_name1[2:] + " " + str(np.round(confidence_score1 * 100))[:-2] + "%"
    # label_text = f"{output_label}"
    # label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # label_x, label_y = 10, 30
    # cv2.putText(white_img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #
    # output_label = class_name2[2:] + " " + str(np.round(confidence_score2 * 100))[:-2] + "%"
    # label_text = f"{output_label}"
    # label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    # label_x, label_y = 10, 30
    # cv2.putText(white_img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


camera.release()
cv2.destroyAllWindows()
