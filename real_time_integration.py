import cv2
import numpy as np
import tensorflow as tf
import pygame

model = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier('C:\Mini - Project\Drowsiness_detection\haarcascade_eye.xml')

alarm_on = False
closed_eye_frames = 0
closed_eyes_limit = 25

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    if len(eyes) == 0 and alarm_on:
        pygame.mixer.music.stop()
        alarm_on = False

    for (x, y, w, h) in eyes:
        eye_img = gray[y:y+h, x:x+w]
        eye_img = cv2.resize(eye_img, (224, 224))
        eye_img = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2RGB)
        eye_img = np.expand_dims(eye_img, axis=0)
        prediction = model.predict(eye_img)

        if np.any(prediction < 0.4):
            closed_eye_frames += 1
        else:
            closed_eye_frames = 0

        if closed_eye_frames >= closed_eyes_limit:
            alarm_file = "C:\Mini - Project\Drowsiness_detection/music.wav"
            pygame.mixer.init()

            if not alarm_on:
                pygame.mixer.music.load(alarm_file)
                pygame.mixer.music.play()
                alarm_on = True

                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        else:
            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
