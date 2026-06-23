"""
Testskript fuer Aufgabe 1 (evaluate_classifier) und Aufgabe 2 (Trainingsdaten J-R).

Ausfuehren:
    python test_meine_aufgabe.py

Ablauf:
    1. Kamera oeffnet sich fuer jeden Buchstaben
    2. Du nimmst je 3 Aufnahmen auf
    3. Danach wird der Klassifikator automatisch evaluiert
"""

from GestureRecognition.labeling import data_labeling, dataset_building
from GestureRecognition.hmmclassifier import HMMClassifier

# ---------------------------------------------------------------
# SCHRITT 1: Trainingsdaten aufnehmen
# Steuerung im Kamerafenster:
#   LEERTASTE  -> Aufnahme starten
#   LEERTASTE  -> Aufnahme stoppen
#   ESC        -> Aufnahme SPEICHERN
#   andere Taste (z.B. N) -> Aufnahme VERWERFEN
#   Q          -> diesen Buchstaben ueberspringen
# ---------------------------------------------------------------

BUCHSTABEN = ["J", "K", "L"]   # <-- zum Testen erst nur 3 Buchstaben
AUFNAHMEN_PRO_BUCHSTABE = 3    # <-- zum Testen reichen 3

print("=" * 50)
print("SCHRITT 1: Trainingsdaten aufnehmen")
print("=" * 50)
print(f"Buchstaben: {BUCHSTABEN}")
print(f"Aufnahmen pro Buchstabe: {AUFNAHMEN_PRO_BUCHSTABE}")
print()
print("Anleitung:")
print("  1. Kamera oeffnet sich")
print("  2. Forme die Handgeste fuer den Buchstaben")
print("  3. LEERTASTE druecken -> Aufnahme startet (roter Text)")
print("  4. Geste kurz halten oder bewegen (~1-2 Sekunden)")
print("  5. LEERTASTE druecken -> Aufnahme stoppt")
print("  6. ESC druecken -> gespeichert!")
print("  7. Wiederhole bis alle Aufnahmen fertig sind")
print()
input("Enter druecken zum Starten...")

for buchstabe in BUCHSTABEN:
    print(f"\n>>> Buchstabe: {buchstabe} <<<")
    data_labeling(AUFNAHMEN_PRO_BUCHSTABE, buchstabe)

# ---------------------------------------------------------------
# SCHRITT 2: Datensatz zusammenbauen
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("SCHRITT 2: Datensatz bauen")
print("=" * 50)

sequences, labels = dataset_building("data/dataset.pkl")

if len(sequences) == 0:
    print("\nKeine Daten gefunden! Bitte zuerst Aufnahmen machen.")
    exit()

# ---------------------------------------------------------------
# SCHRITT 3: evaluate_classifier testen
# ---------------------------------------------------------------
print("\n" + "=" * 50)
print("SCHRITT 3: Klassifikator evaluieren")
print("=" * 50)

clf = HMMClassifier(n_components=3, n_iter=50)
results = clf.evaluate_classifier(sequences, labels)

print(f"\n>>> Ergebnis: {results['accuracy'] * 100:.1f}% Genauigkeit <<<")
print("evaluate_classifier() funktioniert!")
