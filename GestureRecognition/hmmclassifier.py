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

