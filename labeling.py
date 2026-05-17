import cv2
import mediapipe as mp
import csv
import copy

# Konfiguration
model_path = 'hand_landmarker.task'
csv_path = 'model/keypoint.csv' # Die Datei wird automatisch erstellt

def log_to_csv(label, landmarks):
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        # Wir flachen die Liste der Landmarks (x,y,z) zu einer langen Zeile ab
        writer.writerow([label, *landmarks])

# MediaPipe Setup
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
    
    print("Drücke '0'-'9' um Daten mit dem jeweiligen Label zu speichern. 'q' zum Beenden.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Frame vorbereiten
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        
        # Erkennung
        result = landmarker.detect_for_video(mp_image, timestamp)
        
        # Visualisierung & Speichern
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                # Koordinaten für CSV vorbereiten (Normalisierung auf 0.0 - 1.0)
                data_row = []
                for lm in landmarks:
                    data_row.extend([lm.x, lm.y, lm.z])
                
                # Tastenabfrage zum Speichern
                key = cv2.waitKey(1) & 0xFF
                if ord('0') <= key <= ord('9'):
                    label = chr(key)
                    log_to_csv(label, data_row)
                    print(f"Geste {label} gespeichert!")
                elif key == ord('q'):
                    break

        cv2.imshow('Labeling Studio', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()