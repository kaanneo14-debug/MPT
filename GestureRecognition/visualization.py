import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt


def visualize_dataset(data_dir="processed_data/train"):
    """
    Visualisiert die prozessierten Trajektorien der Gesten.
    Ziel: Optische Verifikation der Datenqualität und Identifikation von Ausreißern.
    """
    if not os.path.exists(data_dir):
        print(
            f"Fehler: Verzeichnis '{data_dir}' existiert nicht. Führe zuerst dataset_building() aus."
        )
        return

    # Wir betrachten alle Klassen, um das "Big Picture" zu wahren
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)

    # Raster für die Subplots berechnen
    cols = 3
    rows = int(np.ceil(num_classes / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    # Sicherstellen, dass axes immer ein flaches Array ist, auch bei wenig Klassen
    if num_classes == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        samples = os.listdir(cls_dir)

        ax = axes[i]
        ax.set_title(f"Klasse: {cls} ({len(samples)} Samples)")

        for sample in samples:
            # Lade die prozessierten Differenzvektoren
            traj_diff = np.load(os.path.join(cls_dir, sample))

            # Rekonstruktion der absoluten Form via cumsum (Startpunkt bei 0,0)
            traj_shape = np.vstack([[0, 0], np.cumsum(traj_diff, axis=0)])

            # y-Achse invertieren, da Bildkoordinaten (OpenCV/Webcam) y nach unten definieren,
            # Matplotlib jedoch standardmäßig y nach oben plottet.
            ax.plot(traj_shape[:, 0], -traj_shape[:, 1], alpha=0.3, linewidth=1.5)

            # Optional: Den Startpunkt als kleinen Punkt markieren (hilft bei der Analyse der Zeichenrichtung)
            ax.scatter(0, 0, color="red", s=10, zorder=5)

        ax.axis("equal")  # Wichtig, um Verzerrungen der Proportionen zu vermeiden
        ax.grid(True, linestyle="--", alpha=0.6)

    # Verbleibende leere Subplots ausblenden
    for j in range(num_classes, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_dataset()
    """
    TODO: Visualisierung des eigenen Datensatzes

    Ziel:
    -----
    Entwickle eine Möglichkeit, deinen aufgenommenen Datensatz visuell zu
    inspizieren und zu verstehen.

    Warum ist das wichtig?
    ----------------------
    - Du musst nachvollziehen können, was dein Modell eigentlich „sieht“
    - Fehler im Datensatz lassen sich visuell oft sofort erkennen
    - Qualität der Daten ist entscheidend für die Modellperformance

    Anforderungen / Ideen:
    ----------------------
    - Lade deinen Trainingsdatensatz
    - Visualisiere mehrere Sequenzen pro Klasse
    - Stelle sicher, dass:
        - unterschiedliche Gesten klar unterscheidbar sind
        - Sequenzen sinnvoll aussehen (keine Ausreißer, keine leeren Daten)

    .. tip::
       Ein einfacher Ansatz:
         - Plotte Trajektorien (z. B. x/y-Koordinaten)
         - Zeige mehrere Beispiele pro Klasse übereinander

    .. note::
       Du kannst selbst entscheiden:
         - Wie viele Sequenzen du anzeigst
         - Welche Features du visualisierst
         - Ob du interaktive Elemente einbaust

    .. tip::
       Interaktivität (z. B. Klick auf eine Sequenz) kann hilfreich sein,
       um einzelne Beispiele genauer zu untersuchen.

    Abgabe:
    -------
    - Du musst in der Lage sein, deinen Datensatz visuell zu präsentieren
    - Du solltest erklären können:
        - Wie unterscheiden sich die Klassen?
        - Gibt es problematische Beispiele?

    Erweiterung (optional):
    -----------------------
    - Mittelwerte oder typische Sequenzen pro Klasse darstellen
    - Ausreißer automatisch erkennen
    """
    pass

def evaluate_classifier():
    """Trainiert den HMMClassifier und plottet Accuracy + Konfusionsmatrix auf den Testdaten."""
    from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score

    from .hmmclassifier import HMMClassifier

    # Modell trainieren (train/test bereits durch dataset_building() angelegt)
    clf = HMMClassifier()
    clf.fit()

    # Auf Testdaten evaluieren
    true_labels = []
    pred_labels = []

    for true_label in os.listdir("processed_data/test"):
        test_dir = f"processed_data/test/{true_label}"
        for sample_file in os.listdir(test_dir):
            traj = np.load(f"{test_dir}/{sample_file}")
            true_labels.append(true_label)
            pred_labels.append(clf.predict(traj))

    correct = int(accuracy_score(true_labels, pred_labels, normalize=False))
    accuracy = correct / len(true_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%  ({correct}/{len(true_labels)})")

    # Konfusionsmatrix plotten
    classes = sorted(set(true_labels) | set(pred_labels))
    size = min(14, max(6, len(classes) * 0.45))
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay.from_predictions(
        true_labels, pred_labels, labels=classes, cmap="Blues", colorbar=True, ax=ax,
    )
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title(f"Konfusionsmatrix  |  Accuracy: {accuracy * 100:.1f}%")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.show()


def replay_recordings(label=None):
    """Spielt aufgezeichnete Rohdaten Beispiel für Beispiel ab (ESC im Fenster = weiter zum nächsten)."""
    labels = [label] if label else sorted(os.listdir("datasets"))
    for lbl in labels:
        for sample in sorted(os.listdir(f"datasets/{lbl}")):
            path = f"datasets/{lbl}/{sample}"
            print(f"Replay: {path}  (ESC im Fenster zum Weiterspringen)")
            subprocess.run([
                sys.executable, "-m", "GestureRecognition.demo",
                "--mode", "replay",
                "--recorder.file", path,
            ])