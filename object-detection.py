import cv2
import mediapipe as mp
import time

mp_objectron = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

pTime = 8
cTime = 0


cap = cv2.VideoCapture(0)

with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=2,
                            min_detection_confidence=.05,
                            min_tracking_confidence=0.8,
                            model_name='Chair') as objectron:
    while cap.isOpened():

        success, img = cap.read()

        start = time.time()

        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False
        results = objectron.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                mp_drawing.draw_landmarks(img, detected_object.landmarks_2d, mp_objectron.BOX_CONNECTIONS)
                mp_drawing.draw_axis(img, detected_object.rotation, detected_object.translation)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (18, 78),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow('MediaPipe Objectron', img)

        if cv2.waitKey(5) & 0xFF == 27:
            break