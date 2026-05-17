import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# 1. Daten laden
if not os.path.exists('model/keypoint.csv'):
    print("Fehler: keypoint.csv nicht gefunden!")
    exit()

df = pd.read_csv('model/keypoint.csv', header=None)
X = df.iloc[:, 1:].values  # Alle Spalten außer der ersten (die Koordinaten)
y = df.iloc[:, 0].values   # Die erste Spalte (die Labels)

# 2. Daten aufteilen (Training & Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Modell definieren und trainieren
print("Training läuft...")
model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, activation='relu')
model.fit(X_train, y_train)

# 4. Genauigkeit prüfen
y_pred = model.predict(X_test)
print(f"Genauigkeit: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# 5. Modell speichern
with open('model/gesture_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Modell gespeichert unter model/gesture_classifier.pkl")