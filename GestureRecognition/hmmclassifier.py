import pickle
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import train_test_split


class HMMClassifier:
    """
    HMM-basierter Klassifikator für zeitliche Gesten-Sequenzen.

    Pro Klasse wird ein separates GaussianHMM trainiert.
    Klassifikation erfolgt über den höchsten Log-Likelihood-Score.
    """

    def __init__(self, n_components=4, n_iter=100):
        self.n_components = n_components
        self.n_iter = n_iter
        self.models = {}
        self.classes_ = []

    def fit(self, sequences, labels):
        """
        Trainiere ein separates HMM pro Klasse.

        Parameters
        ----------
        sequences : list of np.ndarray
            Liste von Sequenzen, jede mit Shape (T_i, n_features).
        labels : list
            Klassenlabel pro Sequenz.

        Returns
        -------
        self
        """
        label_to_seqs = {}
        for seq, label in zip(sequences, labels):
            label_to_seqs.setdefault(label, []).append(seq)

        self.classes_ = sorted(label_to_seqs.keys())

        for label, seqs in label_to_seqs.items():
            X = np.concatenate(seqs, axis=0)
            lengths = [len(s) for s in seqs]
            model = GaussianHMM(
                n_components=self.n_components,
                covariance_type="diag",
                n_iter=self.n_iter,
                random_state=42,
            )
            model.fit(X, lengths)
            self.models[label] = model

        return self

    def decision_function(self, sequences):
        """
        Berechne Log-Likelihood-Scores für jede Klasse.

        Parameters
        ----------
        sequences : list of np.ndarray
            Liste von Sequenzen.

        Returns
        -------
        scores : np.ndarray, shape (n_sequences, n_classes)
        """
        scores = []
        for seq in sequences:
            seq_scores = []
            for label in self.classes_:
                try:
                    score = self.models[label].score(seq)
                except Exception:
                    score = -np.inf
                seq_scores.append(score)
            scores.append(seq_scores)
        return np.array(scores)

    def predict(self, sequences):
        """
        Weise jeder Sequenz das Label mit dem höchsten Score zu.

        Parameters
        ----------
        sequences : list of np.ndarray

        Returns
        -------
        labels : list
            Vorhergesagte Klassenlabels.
        """
        scores = self.decision_function(sequences)
        indices = np.argmax(scores, axis=1)
        return [self.classes_[i] for i in indices]

    def evaluate_classifier(self, sequences, labels, test_size=0.2, random_state=42):
        """
        Train-Test-Split, Training und Evaluation des Klassifikators.

        Parameters
        ----------
        sequences : list of np.ndarray
            Alle verfügbaren Sequenzen.
        labels : list
            Zugehörige Klassenlabels.
        test_size : float
            Anteil der Testdaten (Standard: 0.2).
        random_state : int
            Seed für Reproduzierbarkeit.

        Returns
        -------
        dict mit:
            - "accuracy": float
            - "predictions": list
            - "true_labels": list
            - "confusion_matrix": np.ndarray
            - "classes": list
        """
        indices = list(range(len(sequences)))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )

        train_seqs = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_seqs = [sequences[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]

        self.fit(train_seqs, train_labels)
        pred_labels = self.predict(test_seqs)

        correct = sum(p == t for p, t in zip(pred_labels, test_labels))
        accuracy = correct / len(test_labels)

        classes = sorted(set(test_labels) | set(pred_labels))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        n = len(classes)
        confusion = np.zeros((n, n), dtype=int)
        for true, pred in zip(test_labels, pred_labels):
            confusion[class_to_idx[true], class_to_idx[pred]] += 1

        print(f"Accuracy: {accuracy * 100:.2f}%  ({correct}/{len(test_labels)})")
        print(f"Klassen: {classes}")
        print("Konfusionsmatrix (Zeile=True, Spalte=Pred):")
        print(confusion)

        return {
            "accuracy": accuracy,
            "predictions": pred_labels,
            "true_labels": test_labels,
            "confusion_matrix": confusion,
            "classes": classes,
        }

    def save(self, path):
        """Speichere das trainierte Modell als pickle."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """Lade ein gespeichertes Modell."""
        with open(path, "rb") as f:
            return pickle.load(f)
