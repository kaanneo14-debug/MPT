from SignalHub import Module, get_nested_key, GALY
from collections import deque

class TrailMarker(Module):
    def __init__(self, outputSignal="trailmarker"):
        super().__init__(
            inputSignals=["config", "detector"],
            outputSchema={"type": "object", "properties": {outputSignal: {}}},
            name="trailmarker",
        )

    def start(self, data):
        # Parameter aus config einlesen
        config = data["config"]
        self.finger_idx = get_nested_key("preprocessor.finger_idx", config)  # Welcher Landmark zum zeichnen benutzt wird
        self.max_lost = get_nested_key("preprocessor.max_lost", config)  # Nach wie vielen frames ohne erkennung, die Zeichnung abbricht
        self.buffer_size = get_nested_key("preprocessor.buffer_size", config)  # Max-Länge der Trajektorie

        # Datentyp zum aufzeichnen der Trajektorie erstellen
        self.trajectory = deque(maxlen=self.buffer_size)
        self.lost_frames_counter = 0
        self.galy = None
        return {}

    def step(self, data):
        input_data=data["detector"] #hier werden die detektierten hände plus landmarks geladen (für einen schritt)

        if input_data is None or len(input_data.hand_landmarks) == 0:
          self.galy = GALY(data["webcam"])

        if input_data is None:        # wenn nichts detektiert wird ist die funktion vorbei
          self.lost_frames_counter+=1 # der lost frames counter geht dann einen hoch

          if self.lost_frames_counter > self.max_lost: # wenn Max lost frames überschritten wird , wird die trajectory neu angefangen und der counter zurückgesetzt
             self.trajectory.clear()
             self.lost_frames_counter=0
             
          return {}
        
        self.lost_frames_counter = 0
               
        mark = input_data.hand_landmarks[0][self.finger_idx]  # landmark von zeichnenden finger definiert
        self.trajectory.append((mark.x, mark.y))          #position dieser landmark gespeichert im trajectory
        traj = self.trajectory

        if len(traj) > 1:             # nur linien malen wenn mehr als 1 punkt vorhanden
          x1, y1 = traj[-2]            # hier wird die position der vorletzte landmark definiert
          x2, y2 = traj[-1]            # hier die position des letzten eingetragenen landmark

          self.galy.line(x1, y1, x2, y2)    # hier wird eine linie von der letzten zur vorletzten landmark im galy gespeichert

        return {"galy": self.galy}

    def stop(self, data):
        pass