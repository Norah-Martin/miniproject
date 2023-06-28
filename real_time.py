import cv2
import numpy as np
import tensorflow as tf
import pygame

model = tf.keras.models.load_model(r'C:\Mini - Project\Drowsiness_detection\dataset\model\model_weights.h5')


#model = tf.keras.models.load_model('C:\Mini - Project\Drowsiness_detection\dataset\model\model_weights.h5')

cap = cv2.VideoCapture(0)

eye_cascade = cv2.CascadeClassifier('C:\Mini - Project\Drowsiness_detection\haarcascade_eye.xml')

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in eyes:
        eye_img = gray[y:y+h, x:x+w]
        eye_img = cv2.resize(eye_img, (224, 224))
        eye_img = np.expand_dims(eye_img, axis=0)
        eye_img = np.expand_dims(eye_img, axis=-1)
        prediction = model.predict(eye_img)

        if prediction < 0.4:
            prediction1 = "drowsy"
        else:
            prediction1 = "not-drowsy"

        alarm_file = "path/to/music.wav"
        pygame.mixer.init()

        if prediction1 == "drowsy":
            pygame.mixer.music.load(alarm_file)
            pygame.mixer.music.play()
            cv2.putText(frame, "****************ALERT!****************", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "****************ALERT!****************", (10, 325),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            while pygame.mixer.music.get_busy():
                pygame.time.wait(1000)
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        pygame.mixer.music.stop()

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
