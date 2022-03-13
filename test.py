from turtle import right
import cv2 # opencv library
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # image_grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # faces = detector(image_grey)
    detectorPaths = {
    "face": "haarcascade_frontalface_default.xml",
    "eyes": "haarcascade_eye.xml",
    "smile": "haarcascade_smile.xml",
    }

    cascade_classifier = cv2.CascadeClassifier(
    f"{cv2.data.haarcascades}haarcascade_frontalface_alt2.xml")

    cascade_classifier2 = cv2.CascadeClassifier(
    f"{cv2.data.haarcascades}haarcascade_eye.xml")

    # cascade_classifier3 = cv2.CascadeClassifier(
    #     f"{cv2.data.haarcascades}haarcascade_smile.xml")
    
    left_eye = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_lefteye_2splits.xml")
    
    rigth_eye = cv2.CascadeClassifier(
        f"{cv2.data.haarcascades}haarcascade_rigtheye_2splits.xml")

    detected_objects = cascade_classifier.detectMultiScale(frame, minSize=(28, 28))


    if len(detected_objects) != 0:
        for key, (x, y, width, height) in enumerate(detected_objects):
            cv2.rectangle(frame, (x, y),
                (x + height, y + width),
                (0, 255, 0), 2)

        crop_img = frame[y:y+height, x:x+width]

        detected_eyes = cascade_classifier2.detectMultiScale(crop_img, minSize=(28, 28))

        # detected_left_eye = left_eye.detectMultiScale(crop_img, minSize=(28, 28))

        # print(len(detected_left_eye), "=====")
        # for key, (x, y, width, height) in enumerate(detected_left_eye):
        #     if key == 0:
        #         cv2.rectangle(crop_img, (x, y),
        #             (x + height, y + width),
        #             (0, 255, 0), 2)
        

        if len(detected_eyes) == 2:
            print(len(detected_eyes))

            plus_x = 25
            plus_y = 25

            x_1 = detected_eyes[0][0] - plus_x
            y_1 = detected_eyes[0][1] - plus_y
            width_1 = detected_eyes[0][2]
            height_1 = detected_eyes[0][3]

            x_2 = detected_eyes[1][0]
            y_2 = detected_eyes[1][1]
            width_2 = detected_eyes[1][2]
            height_2 = detected_eyes[1][3]

            isSwaped = x_1 > x_2

            x_total = width_2 + x_2
            y_total = y_1 + height_1


            # print(detected_eyes[0][0], "=====")
            # for key, (x, y, width, height) in enumerate(detected_eyes):

            #     if key <= 1:
            #         if key == 0:
            #             x_1 = x - plus_x
            #             y_1 = y - plus_y
            #             y_total = y + height

            #         elif key == 1:
            #             x_total = width + x
            #             # y_total = y_total + y
            #             # width_total = width_total + width
            #             # height_total = height_total + height
            if isSwaped:
                cv2.rectangle(
                    crop_img,
                    (x_2, y_2),
                    (x_total - plus_x, y_total - plus_y),
                    (0, 255, 0),
                    2
                )
            else:
                cv2.rectangle(
                    crop_img,
                    (x_1, y_1),
                    (x_total + plus_x, y_total + plus_y),
                    (0, 255, 0),
                    2
                )

        # detected_mouth = cascade_classifier3.detectMultiScale(crop_img, minSize=(28, 28))
        # if len(detected_mouth) != 0:
        #     for key, (x, y, width, height) in enumerate(detected_mouth):
        #         cv2.rectangle(crop_img, (x, y),
        #             (x + height, y + width),
        #             (0, 255, 0), 2)


        #  print(crop_img, "===== key ====")
    if cv2.waitKey(1) == ord('q'):
        break
    cv2.imshow('frame', frame)

cap.release()
cv2.destroyAllWindows()