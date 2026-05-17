import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Modell laden
with open('model/gesture_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

labels_dict = {0: 'Faust', 1: 'Offen', 2: 'Eins', 3: 'Peace', 4: 'Daumen'}

# 2. MediaPipe Setup (Tasks API)
model_path = 'hand_landmarker.task'
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Erkennung
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                # Daten für Modell vorbereiten
                data_row = []
                for lm in landmarks:
                    data_row.extend([lm.x, lm.y, lm.z])
                
                # Vorhersage
                prediction = model.predict([data_row])
                gesture_name = labels_dict.get(int(prediction[0]), "Unbekannt")
                
                # Text auf das Bild schreiben
                cv2.putText(frame, gesture_name, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow('MPT Live Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()