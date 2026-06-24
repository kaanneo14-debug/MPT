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

BUCHSTABEN = list("JKLMNOPQR")  # alle 9 Buchstaben J bis R
AUFNAHMEN_PRO_BUCHSTABE = 5    # 5 Aufnahmen pro Buchstabe

print("=" * 50)
print("SCHRITT 1: Trainingsdaten aufnehmen")
print("=" * 50)
print(f"Buchstaben: {BUCHSTABEN}")
print(f"Aufnahmen pro Buchstabe: {AUFNAHMEN_PRO_BUCHSTABE}")
print()
print("Anleitung:")
print("  1. Kamera oeffnet sich fuer jeden Buchstaben")
print("  2. Forme die Handgeste")
print("  3. LEERTASTE -> Aufnahme startet (roter Text)")
print("  4. ~1-2 Sekunden halten")
print("  5. LEERTASTE -> stoppt und speichert automatisch")
print("  6. Wiederhole bis (5/5), dann geht es zum naechsten Buchstaben")
print("  7. Q oder ESC = diesen Buchstaben abbrechen")
print()
antwort = input("Neue Aufnahmen machen? (j/n): ").strip().lower()

if antwort == "j":
    for buchstabe in BUCHSTABEN:
        print(f"\n>>> Buchstabe: {buchstabe} <<<")
        data_labeling(AUFNAHMEN_PRO_BUCHSTABE, buchstabe)
else:
    print("Aufnahme uebersprungen, nutze vorhandene Daten.")

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
