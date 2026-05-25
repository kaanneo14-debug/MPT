import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from SignalHub import GALY, bgr, get_nested_key, Module

mp_hand = mp.tasks.vision.HandLandmarksConnections


def draw_hand_landmarks(hand_landmarks, galy: GALY):
    lm = {
        "thumb":         {"color": bgr("#0000FF")},
        "index_finger":  {"color": bgr("#00FF00")},
        "middle_finger": {"color": bgr("#FF0000")},
        "ring_finger":   {"color": bgr("#00FFFF")},
        "pinky_finger":  {"color": bgr("#FF00FF")},
        "palm":          {"color": bgr("#C8C8C8")},
    }
    x = np.inf
    y = np.inf
    for key in lm.keys():
        pts = set()
        for conn in getattr(mp_hand, f"HAND_{key.upper()}_CONNECTIONS"):
            start = (hand_landmarks[conn.start].x,
                    hand_landmarks[conn.start].y)
            end = (hand_landmarks[conn.end].x,
                hand_landmarks[conn.end].y)
            x = min(x, start[0], end[0])
            y = min(y, start[1], end[1])
            galy.line(start, end, lm[key]["color"], 2)
            pts.update([conn.start, conn.end])
        for pt in pts:
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 5, (255,255,255), 1)
            galy.circle((hand_landmarks[pt].x, hand_landmarks[pt].y), 4, lm[key]["color"], -1)


class HandDetector(Module):
    def __init__(self, outputSignal="detector"):
        super().__init__(
            inputSignals=["config", "webcam"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="detector",
        )

    def start(self, data):
        # 1. Basis-Optionen festlegen: Wo liegt das trainierte Modell?
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        
        # 2. Spezifische Optionen für die Handdetektion konfigurieren
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2, # Relevant, falls der Prüfer zweihändige Gesten testet
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 3. Das Modell in den Arbeitsspeicher laden und an die Instanz (self) binden
        self.detector = vision.HandLandmarker.create_from_options(options)        
        return {}

    def step(self, data):

        bild = data["webcam"]

        bild2 = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)
        bild3 = mp.Image(image_format=mp.ImageFormat.SRGB, data=bild2)

        result = self.detector.detect(bild3)

        galy = GALY()

        for i in range(len(result.hand_landmarks)):
            hand = result.hand_landmarks[i]
            draw_hand_landmarks(hand, galy)

        return {"detector": result, "galy": galy}

    def stop(self, data):
        pass
