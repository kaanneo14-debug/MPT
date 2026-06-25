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



def data_labeling(times: int, label: str):
    """
    TODO: data_labeling: Datenerfassung für Gesten (SignalHub)

    Ziel:
    -----
    Implementiere eine Funktion, mit der Trainingsdaten für eine bestimmte
    Geste aufgenommen und gespeichert werden können.

    Anforderungen / Ideen:
    ----------------------

    1. Aufnahme starten

       - Starte SignalHub über einen Subprocess
       - Übergib einen Dateipfad für die Aufnahme
       - Überlege, welche Module aufgenommen werden sollen
       - Nimm entsprechende Änderungen in der ``config.yaml`` vor

    root = os.getcwd()
    data_path = os.path.join(root, "data")

    2. Interaktive Steuerung (optional)

       - Implementiere eine einfache Benutzerinteraktion:
         - Aufnahme speichern
         - Aufnahme verwerfen
         - Programm beenden

    .. tip::

       Die Funktion ``getch()`` (Aus dem Modul Linux :mod:`getch` oder bei Windows :mod:`msvcrt`) ist sehr hilfreich, um einzelne Tastendrücke
       direkt auszulesen (ohne Enter). Damit kannst du dir ein schnelles
       Labeling-Interface bauen.

       Beispiel:

       .. code-block:: text

           ESC → speichern
           andere Taste → verwerfen

    3. Daten sichten und bereinigen

       - Lade die aufgenommenen Daten
       - Überlege:
         - Welche Teile sind relevant?
         - Welche Frames sind leer oder unbrauchbar?
         - Sollten gewisse Sequenzen evtl. gar nicht benutzt werden?
       - Entferne unnötige Anteile (z. B. keine erkannte Hand am Anfang/Ende)

    4. Speicherung

       - Speichere Daten strukturiert nach Labels (z. B. Ordnerstruktur)
       - Jede Aufnahme sollte einzeln gespeichert werden

    .. note::

       Die konkrete Umsetzung (Dateiformat, Struktur, Ablauf) ist bewusst offen.
       Entwickle ein System, das für dich sinnvoll ist und sich gut weiterverarbeiten lässt.

    .. warning::

       Ziel ist nicht nur, dass es „funktioniert“, sondern ein sauberer und
       effizienter Workflow für Datensammlung.

    Parameters
    ----------
    times : int
        Wie viele Aufnahmen gemacht werden sollen.
        Kann frei angepasst werden (z. B. Endlosschleife oder interaktive Steuerung).

    label : str
        Name der Geste / Klasse.
        Kann ebenfalls frei gestaltet werden (z. B. dynamische Labels, mehrere Klassen gleichzeitig).
    """
    counter = 0
    for _ in range(times):
      # Ordner erstellen, falls nicht vorhanden
      oberordner = rf"datasets/{label}"
      os.makedirs(oberordner, exist_ok=True)

      # Richtigen filenamen rausifnden mit regex
      max_index = -1
      for datei in os.listdir(oberordner):
         match = re.search(r"(\d+)\.pkl$", datei)
         if match:
               index = int(match.group(1))
               max_index = max(max_index, index)
      neuer_index = max_index + 1
      name = f"{label}_{neuer_index}.pkl"
      zielpfad = os.path.join(oberordner, name)

      # Pipeline starten
      prozess = subprocess.Popen([
                  sys.executable,
                  "GestureRecognition/demo.py",
                  "--mode", "record",
                  "--recorder.file", zielpfad],
                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
               
      # Beenden bei Tastendruck
      print("ESC zum Beenden der AUfnahme")      
      prozess.wait()

        
      eingabe = input("Zum verwerfen N drücken. Ansonsten beliebige Taste")  # input ist os unabhängig
   

      # Verwerfen
      if eingabe == "n":
         os.remove(zielpfad)
         continue
      counter += 1
      if counter == times:
          return





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
    #dataset_building()
    data_labeling(5, "W")