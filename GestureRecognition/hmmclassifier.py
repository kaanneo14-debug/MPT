import os
import numpy as np
import pickle
from hmmlearn import hmm

class HMMClassifier:
    """
    TODO: Implementiere einen HMM-basierten Klassifikator

    Ziel:
    -----
    Entwickle einen Klassifikator, der zeitliche Sequenzen mit Hilfe von
    Hidden-Markov-Modellen (HMMs) klassifiziert. Für HMMs können libraries wie
    :mod:`hmmlearn` benutzt werden

    Grundidee:
      ----------
    - Trainiere ein Modell pro Klasse
    - Bewerte neue Sequenzen anhand der Likelihood unter jedem Modell
    - Wähle die Klasse mit der höchsten Wahrscheinlichkeit

    .. note::
       Wie genau deine Modelle aussehen (z. B. Anzahl Zustände, Features,
       Initialisierung etc.) ist bewusst nicht vorgegeben.

    Wichtige Designentscheidungen:
    ------------------------------
    - Wie strukturierst du deine Trainingsdaten?
    - Wie repräsentierst du Sequenzen?
    - Wie verbindest du mehrere Sequenzen mit Labels?

    Speicherung:
    ------------
    Du solltest dir überlegen:
    - Wie speicherst du dein trainiertes Modell?
    - Wie lädst du es später wieder?
    - Welche Informationen müssen persistiert werden (z. B. Klassen, Modelle)?

    .. tip::
       ``pickle`` ist eine einfache Möglichkeit, Modelle zu speichern.
       Alternativ kannst du auch eigene Formate definieren.

    Evaluation:
    -----------
    Für sinnvolles Training solltest du unbedingt:
    - eine eigene ``train_test_split``-Logik implementieren
    - Trainings- und Testdaten sauber trennen

    .. warning::
       Wenn du Training und Test nicht trennst, sind deine Ergebnisse nicht aussagekräftig.

    Erweiterung (optional):
    -----------------------
    - Implementiere eine Grid Search für Hyperparameter
      (z. B. Anzahl Zustände, Modellstruktur)
    - Vergleiche verschiedene Modellkonfigurationen

    """

    def fit(self):
        """
        TODO: Trainiere den Klassifikator

        Ziel:
        -----
        Trainiere ein separates HMM für jede Klasse basierend auf den
        gegebenen Sequenzen.


        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Daten so, dass du pro Klasse alle Sequenzen bekommst
        - Trainiere ein Modell pro Klasse
        - Speichere die trainierten Modelle intern

        .. tip::
           Überlege dir eine sinnvolle Datenstruktur wie:
           ``label -> (Daten, Sequenzlängen)``

        .. note::
           Die konkrete Umsetzung ist offen:
            - Wie genau du Daten aufteilst
            - Wie du dein Modell initialisierst
            - Welche Hyperparameter du verwendest

        .. warning::
           Achte darauf, dass:
            - ``lengths`` zu ``X`` passen
            - Labels korrekt zu Sequenzen zugeordnet sind

        Erweiterung:
        ------------
        - Experimentiere mit verschiedenen Modellgrößen
        - Nutze eine Grid Search zur Optimierung
        - Verwende ein separates Testset zur Evaluation

        Returns
        -------
        self
        """
        # Vorbereitung
        os.makedirs("models", exist_ok=True)
        # Iteriere oberordner in processed_data | Für jedes Label
        for oberordner in os.listdir("processed_data/train"):
            os.makedirs(f"models/{oberordner}", exist_ok=True)
            X = np.empty((0, 2))
            lengths = []
            for sample in os.listdir(f"processed_data/train/{oberordner}"):
                traj = np.load(f"processed_data/train/{oberordner}/{sample}")
                # Sequenzen konkatinieren
                X = np.concatenate([X, traj], axis=0)
                # Längen konkatinieren
                lengths.append(traj.shape[0])
            # Lade das HMM
            model = hmm.GaussianHMM(n_components=7, covariance_type="diag", n_iter=50)
            # Trainiere das HMM
            model.fit(X, lengths)
            # Speichere das Model in models
            with open(f"models/{oberordner}/{oberordner}.pkl", "wb") as f:
                pickle.dump(model, f)

    def decision_function(self, traj):
        """
        TODO: Berechne Scores für jede Klasse

        Ziel:
        -----
        Berechne für jede Eingabesequenz einen Score pro Klasse
        (z. B. Log-Likelihood unter jedem Modell).

        Anforderungen / Ideen:
        ----------------------
        - Zerlege die Eingabe in einzelne Sequenzen
        - Berechne für jede Sequenz:
            Score unter jedem Klassenmodell
        - Gib eine Struktur zurück wie:
            ``(n_sequences, n_classes)``

        .. tip::
           Die meisten HMM-Implementierungen bieten eine
           ``score``-Funktion für Likelihoods.

        .. note::
           Du entscheidest selbst:
            - Welcher Score verwendet wird
            - Wie du mehrere Sequenzen behandelst

        .. warning::
           Stelle sicher, dass:
            - Die Reihenfolge der Klassen konsistent ist
            - Scores vergleichbar sind

        Returns
        -------
        scores : array-like
            Score pro Sequenz und Klasse
        """
        scores = []
        # Für jedes Klassenmodell
        for oberordner in os.listdir("models"):
            for model_file in os.listdir(f"models/{oberordner}"):
                # Lade das Model
                with open(f"models/{oberordner}/{model_file}", "rb") as f:
                    model = pickle.load(f)
                # Berechne den Score
                score = model.score(traj)
                scores.append( (oberordner, score) )
        return scores


    def predict(self, traj):
        """
        TODO: Sage Klassenlabels voraus

        Ziel:
        -----
        Weise jeder Eingabesequenz ein Label zu.

        Anforderungen / Ideen:
        ----------------------
        - Nutze deine ``decision_function``
        - Wähle für jede Sequenz die Klasse mit bestem Score

        .. tip::
           Typischerweise:
           ``argmax über Klassen``

        .. note::
           Achte darauf, dass:
            - Klassenreihenfolge konsistent ist
            - Rückgabewerte klar interpretierbar sind

        Erweiterung:
        ------------
        - Gib zusätzlich Unsicherheiten oder Scores zurück
        - Implementiere Top-k Vorhersagen

        Returns
        -------
        labels : list
            Vorhergesagte Labels
        """
        max_score = float("-inf")
        scores = self.decision_function(traj)
        for label, score in scores:
            if score > max_score:
                max_score = score
                max_label = label
        return max_label

    def evaluate_classifier(self):
        """
        Teilt die Daten aus processed_data/<label>/ automatisch in Train (80%)
        und Test (20%), trainiert den Klassifikator und berechnet die Accuracy.

        Returns
        -------
        dict mit:
            - "accuracy": float
            - "correct": int
            - "total": int
            - "results": list of (true_label, predicted_label)
        """
        import random
        import shutil

        # Train/Test-Unterordner anlegen und befüllen
        for split in ("train", "test"):
            if os.path.exists(f"processed_data/{split}"):
                shutil.rmtree(f"processed_data/{split}")
            os.makedirs(f"processed_data/{split}")

        for label in os.listdir("processed_data"):
            if label in ("train", "test"):
                continue
            samples = sorted([
                f for f in os.listdir(f"processed_data/{label}")
                if f.endswith(".npy")
            ])
            random.shuffle(samples)
            split_idx = max(1, int(len(samples) * 0.8))
            train_samples = samples[:split_idx]
            test_samples = samples[split_idx:] if len(samples) > 1 else samples[:1]

            for split, files in (("train", train_samples), ("test", test_samples)):
                dest = f"processed_data/{split}/{label}"
                os.makedirs(dest, exist_ok=True)
                for f in files:
                    shutil.copy(f"processed_data/{label}/{f}", f"{dest}/{f}")

        # Trainieren
        self.fit()

        # Evaluieren
        correct = 0
        total = 0
        results = []

        for true_label in os.listdir("processed_data/test"):
            test_dir = f"processed_data/test/{true_label}"
            for sample_file in os.listdir(test_dir):
                traj = np.load(f"{test_dir}/{sample_file}")
                predicted_label = self.predict(traj)
                results.append((true_label, predicted_label))
                total += 1
                if predicted_label == true_label:
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0
        print(f"Accuracy: {accuracy * 100:.2f}%  ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results,
        }


if __name__ == "__main__":
    model = HMMClassifier()
    results = model.evaluate_classifier()
    print(results)