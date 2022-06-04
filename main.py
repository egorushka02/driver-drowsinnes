
#import libraries
import numpy as np
import cv2
from keras.models import load_model

eye_detect = load_model('models/eye_status_detection.h5')
yawn_detect = load_model('models/yawn_detection.h5')


count = 0
score = 0
thicc = 2
leye_status = ""
reye_status = ""
yawn_status = ""
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

cap = cv2.VideoCapture(0)

faces = cv2.CascadeClassifier('cascades/faces.xml')
right_eye = cv2.CascadeClassifier('cascades/right_eye.xml')
left_eye = cv2.CascadeClassifier('cascades/left_eye.xml')

while True:
    succes, img = cap.read()
    height, width = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    result_face = faces.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1)
    face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (256, 256))
    face = face / 255
    face = face.reshape(256, 256, -1)
    face = np.expand_dims(face, axis=0)
    yawn_pred = yawn_detect.predict(face)
    if (yawn_pred[0,0]<yawn_pred[0,1]):
        cv2.rectangle(img, (0, 0), (width, height), (0, 255, 255), 20)
        cv2.putText(img, "Yawn detect", (10, height - 40), font, 1,
                    (0, 0, 0), 1, cv2.LINE_AA)

    for (x, y, w, h) in result_face:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), thickness=2)


    result_left_eye = left_eye.detectMultiScale(gray)
    for (x, y, w, h) in result_left_eye:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
        leye = img[y:y + h, x:x + w]
        count += 1
        leye = cv2.cvtColor(leye, cv2.COLOR_BGR2GRAY)
        leye = cv2.resize(leye, (256, 256))
        leye = leye / 255
        leye = leye.reshape(256, 256, -1)
        leye = np.expand_dims(leye, axis=0)
        leye_pred = eye_detect.predict(leye)
        if (leye_pred[0,0]>leye_pred[0,1]):
            leye_status = "Close"
        else:
            leye_status = "Open"

    result_right_eye = right_eye.detectMultiScale(gray)
    for (x, y, w, h) in result_right_eye:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=1)
        reye = img[y:y + h, x:x + w]
        count += 1
        reye = cv2.cvtColor(reye, cv2.COLOR_BGR2GRAY)
        reye = cv2.resize(reye, (256, 256))
        reye = leye / 255
        reye = leye.reshape(256, 256, -1)
        reye = np.expand_dims(reye, axis=0)
        reye_pred = eye_detect.predict(leye)
        if (reye_pred[0, 0] > reye_pred[0, 1]):
            reye_status = "Close"
        else:
            reye_status = "Open"


    if (reye_status == "Close" and leye_status == "Close"):
        score+=1
        cv2.putText(img, "Closed", (10, height-20), font, 1,
                    (0, 0, 0), 1, cv2.LINE_AA)
    else:
        score-=2
        cv2.putText(img, "Open", (10, height - 20), font, 1,
                    (0, 0, 0), 1, cv2.LINE_AA)

    if (score<0):
        score=0
    cv2.putText(img, "Score"+str(score), (100, height-20), font, 1,
                (0, 0, 0), 1, cv2.LINE_AA)
    if (score > 10):
        if (thicc < 16):
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if (thicc < 2):
                thicc = 2
        cv2.rectangle(img, (0, 0), (width, height), (0, 0, 255), thicc*2)

    cv2.imshow('Me', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
        cap.release()
        cv2.destroyAllWindows()
