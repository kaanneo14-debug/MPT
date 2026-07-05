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
import yaml



def data_labeling(times: int, label: str):
    counter = 0
    while True:
      # Ordner erstellen, falls nicht vorhanden
      oberordner = rf"datasets/{label}"
      os.makedirs(oberordner, exist_ok=True)

      # Richtigen filenamen rausifnden mit regex
      max_num = -1
      # alle dateien iterieren
      for datei in os.listdir(oberordner):
         result = re.search(r"(\d+)\.pkl", datei)
         if result:  # falls datei bereits vorhanden
               index = int(result.group(1))
               max_num = max(max_num, index)
      idx = max_num + 1
      file_name = f"{label}_{idx}.pkl"
      zielpfad = os.path.join(oberordner, file_name)

      # Pipeline starten
      prozess = subprocess.Popen([ # neuen Prozess starten (UNterprozess)
                  sys.executable, # pfad zu akt python-interpreter
                  "GestureRecognition/demo.py",
                  "--mode", "record",
                  "--recorder.file", zielpfad])
               
      # Beenden bei Tastendruck
      print("ESC zum Beenden der AUfnahme")      
      prozess.wait()

        
      eingabe = input("Enter druecken zum Speichern, N zum Verwerfen: ").strip().lower()  # input ist os unabhängig


      # Verwerfen
      if eingabe == "n":
         os.remove(zielpfad)
         continue
      counter += 1
      if counter == times:
          return





def dataset_building(output_path="processed_data"):
    # Finger idx extrahieren
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)
    finger_idx = get_nested_key("preprocessor.finger_idx", config)

    os.makedirs(output_path, exist_ok=True) # sicherstellen, dass zielordner existiert

    # iteriere durch alle oberordner in raw-data
    for oberordner in os.listdir("datasets"):

      # Trainingsdaten festlegen
      files = os.listdir(f"datasets/{oberordner}")
      boundary = int(0.8*len(files))
      train_files = files[:boundary] 

      # iteriere durch alle samples in raw data
      for sample in os.listdir(f"datasets/{oberordner}"):

         # vorbereitunng
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
             pos = landmarks[finger_idx]
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

         # SPeichern in richtigem ordner
         if sample in train_files:
            os.makedirs(f"{output_path}/train", exist_ok=True)
            os.makedirs(f"{output_path}/train/{oberordner}", exist_ok=True)
            zielpfad = f"{output_path}/train/{oberordner}/{sample}"
            np.save(zielpfad, traj) 
            
         else:
            os.makedirs(f"{output_path}/test", exist_ok=True)
            os.makedirs(f"{output_path}/test/{oberordner}", exist_ok=True)
            zielpfad = f"{output_path}/test/{oberordner}/{sample}"
            np.save(zielpfad, traj) 


if __name__ == "__main__":
    # Austesten
    dataset_building()
    #data_labeling(30, "I")
