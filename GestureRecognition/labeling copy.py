import os
import subprocess
import sys
import re
import shutil
import signal
import pickle
from SignalHub import get_nested_key
from collections import deque
import numpy as np

def dataset_building(output_path="processed_data"):
    # processed_data Ordner erstellen
    print(os.getcwd())
    os.makedirs(output_path, exist_ok=True)
    # iteriere durch alle oberordner in raw-data
    for oberordner in os.listdir("datasets"):
      # erstelle den oberordner in processed data
      os.makedirs(f"{output_path}/{oberordner}", exist_ok=True)
      # iteriere durch alle samples in raw data
      for sample in os.listdir(f"datasets/{oberordner}"):
         # vorbereitubng
         traj = deque()
         # lade den sample
         with open(f"datasets/{oberordner}/{sample}", "rb") as f:
             file = pickle.load(f)         
         results = file["detector"] # beeinhaltet detector daten pro frame
         # Deque erstellen
         for frame in range(len(results)):
             frame_data = results[frame]
             #1. Nones/leer werte entf 
             if frame_data is None: # #results[frame] kann None sein
                 continue
             if len(frame_data) == 0: #results[frame] kann leer sein
               continue

             landmarks = frame_data["detector"].hand_landmarks  # kann leere liste sein
             if len(landmarks) == 0:
                continue
             landmarks = landmarks[0]
             pos = landmarks[8]
             traj.append((pos.x, pos.y))
         # transformiere die deque/preprocessing
         if len(traj) == 0: # falls ganze aufnahme keine detektion hat
             continue
         
         # In Array umwandeln
         traj = np.array(traj)
         # 2. relativ zum 1.punkt
         traj = traj - traj[0]
         # 3. durch maximalen astand zum ursprung(1.Koordinate)
         scale = np.max(np.linalg.norm(traj, axis=1))
         if scale > 1e-6:  # falls abstand == 0
               traj = traj / scale
         # 4. Verbindungsvektor
         traj = np.diff(traj, axis=0)

         # Zielpfad definieren
         zielpfad = f"{output_path}/{oberordner}/{sample}"
         # SPeichern
         np.save(zielpfad, traj)
         
if __name__ == "__main__":
    # Austesten
    dataset_building()
    #data_labeling(1, "W")