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
        input_data=data["detector"] 
        self.galy=GALY()
       
        if not input_data.hand_landmarks:        
          self.lost_frames_counter+=1 
          for i in range(len(self.trajectory) - 1):
            x1, y1 = self.trajectory[i]
            x2, y2 = self.trajectory[i + 1]
            
            self.galy.line(
                (x1, y1),
                (x2, y2),
                (255, 0, 0),   # rot
                thickness=2
            )
             
          if self.lost_frames_counter > self.max_lost: 
             self.lost_frames_counter=0
             self.trajectory.clear()
          
          return {"galy": self.galy}
        
        self.lost_frames_counter = 0
               
        mark = input_data.hand_landmarks[0][self.finger_idx]
        width = data["config"]["webcam"]["width"]
        height = data["config"]["webcam"]["height"]


        x = int(mark.x * width)
        y = int(mark.y * height)  

        self.trajectory.append((x, y))
        for i in range(len(self.trajectory) - 1):
            x1, y1 = self.trajectory[i]
            x2, y2 = self.trajectory[i + 1]
             
                
            self.galy.line(
            (x1, y1),
            (x2, y2),
            (255, 0, 0),   # rot
            thickness=2
        )
        
        return {"galy": self.galy}
          

    def stop(self, data):
        pass