import cv2 # opencv library
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # image_grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # faces = detector(image_grey)
    cascade_classifier = cv2.CascadeClassifier(
        # f"{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml")
        f"{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml")

    cascade_classifier2 = cv2.CascadeClassifier(
        # f"{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml")
        f"{cv2.data.haarcascades}haarcascade_eye_tree_eyeglasses.xml")

    detected_objects = cascade_classifier.detectMultiScale(frame, minSize=(28, 28))

    if len(detected_objects) != 0:
        for key, (x, y, width, height) in enumerate(detected_objects):
            cv2.rectangle(frame, (x, y),
                (x + height, y + width),
                (0, 255, 0), 2)

            crop_img = frame[y:y+height, x:x+width]

            detected_eyes = cascade_classifier2.detectMultiScale(crop_img, minSize=(28, 28))
            if len(detected_eyes) != 0:
                for key, (x, y, width, height) in enumerate(detected_eyes):
                    cv2.rectangle(crop_img, (x, y),
                        (x + height, y + width),
                        (0, 255, 0), 2)

    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()