def visualize_dataset():
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
    """
    TODO: Evaluation deines Klassifikators

    Ziel:
    -----
    Implementiere eine sinnvolle Auswertung deines Modells auf Testdaten.

    Warum ist das wichtig?
    ----------------------
    - Du brauchst objektive Metriken für die Qualität deines Modells
    - Training allein reicht nicht, entscheidend ist die Generalisierung

    Anforderungen / Ideen:
    ----------------------
    - Lade ein trainiertes Modell
    - Lade Testdaten (getrennt vom Training!)
    - Berechne Vorhersagen
    - Vergleiche Vorhersagen mit Ground Truth

    Metriken:
    ---------
    - Klassifikationsgenauigkeit (Accuracy)
    - Confusion Matrix

    .. tip::
       Eine Confusion Matrix zeigt dir:
         - Welche Klassen gut erkannt werden
         - Wo dein Modell Fehler macht

    .. warning::
       Testdaten dürfen **nicht** aus dem Training stammen!

    Interpretation:
    ---------------
    Du solltest erklären können:
    - Welche Klassen gut funktionieren
    - Welche Klassen verwechselt werden
    - Warum das passieren könnte

    .. note::
       Schlechte Performance liegt oft an:
         - schlechten Trainingsdaten
         - zu wenigen Beispielen
         - ungeeigneten Features

    Erweiterung (optional):
    -----------------------
    - Weitere Metriken (Precision, Recall, F1)
    - Vergleich verschiedener Modelle
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from hmmclassifier import HMMClassifier

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
            predicted = clf.predict(traj)
            true_labels.append(true_label)
            pred_labels.append(predicted)

    # Accuracy berechnen
    correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    total = len(true_labels)
    accuracy = correct / total if total > 0 else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}%  ({correct}/{total})")

    # Konfusionsmatrix berechnen
    classes = sorted(set(true_labels) | set(pred_labels))
    n = len(classes)
    idx = {c: i for i, c in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        matrix[idx[true], idx[pred]] += 1

    # Konfusionsmatrix plotten
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Vorhergesagt")
    ax.set_ylabel("Tatsächlich")
    ax.set_title(f"Konfusionsmatrix  |  Accuracy: {accuracy * 100:.1f}%")

    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    plt.tight_layout()
    plt.show()


def replay_recordings():
    """
    TODO: Exploration und Replay der aufgenommenen Rohdaten

    Ziel:
    -----
    Ermögliche es, aufgenommene Sequenzen erneut abzuspielen
    und qualitativ zu überprüfen.

    Warum ist das wichtig?
    ----------------------
    - Du kannst überprüfen, ob deine Aufnahmen korrekt sind
    - Fehler in der Datenerfassung werden früh sichtbar
    - Du entwickelst ein besseres Verständnis für deine Daten

    Anforderungen / Ideen:
    ----------------------
    - Lade gespeicherte Aufnahmen
    - Spiele diese erneut ab (z. B. über SignalHub / Replay-Modus)
    - Iteriere über verschiedene Labels und Beispiele

    .. tip::
       Besonders hilfreich:
         - Vergleiche mehrere Beispiele derselben Klasse
         - Suche nach inkonsistenten Bewegungen

    .. warning::
       Schlechte oder inkonsistente Aufnahmen führen fast immer zu
       schlechten Modellen. Überprüfe deine Daten frühzeitig!

    Abgabe:
    -------
    - Du solltest zeigen können, wie deine Daten aussehen (Replay)
    - Du solltest erklären können:
        - Welche Beispiele gut sind
        - Welche problematisch sind

    Erweiterung (optional):
    -----------------------
    - Automatisches Filtern schlechter Sequenzen
    - Kombination mit Visualisierung
    """
    pass