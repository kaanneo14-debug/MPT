import cv2
import mediapipe as mp
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path("data/gestures")


def data_labeling(times: int, label: str):
    """
    Nimm Trainingssequenzen für eine Geste auf und speichere sie.

    Steuerung:
        LEERTASTE  – Aufnahme starten / stoppen
        ESC        – gestoppte Aufnahme speichern
        andere     – gestoppte Aufnahme verwerfen
        Q          – Programm beenden

    Jede Aufnahme ist eine zeitliche Sequenz aller 21 Hand-Landmarks
    (je x, y, z) und wird als .pkl-Datei unter data/gestures/<label>/
    gespeichert.

    Parameters
    ----------
    times : int
        Anzahl der zu speichernden Aufnahmen.
    label : str
        Name der Geste / des Buchstabens (z.B. "J").
    """
    data_dir = DATA_DIR / label
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = list(data_dir.glob("*.pkl"))
    recording_idx = len(existing)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    recordings_done = 0
    recording = False
    current_sequence = []

    print(f"\nLabel: '{label}'  |  Ziel: {times} Aufnahmen")
    print("LEERTASTE = Aufnahme starten / stoppen + speichern")
    print("Q oder Fenster schliessen = beenden\n")

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("Labeling")

        while cap.isOpened() and recordings_done < times:
            ret, frame = cap.read()
            if not ret:
                break

            # Fenster-X-Button abfangen
            if cv2.getWindowProperty("Labeling", cv2.WND_PROP_VISIBLE) < 1:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp)

            hand_detected = bool(result.hand_landmarks)

            if recording and hand_detected:
                landmarks = result.hand_landmarks[0]
                frame_features = []
                for lm in landmarks:
                    frame_features.extend([lm.x, lm.y, lm.z])
                current_sequence.append(frame_features)

            # Statusanzeige
            if recording:
                status = f"REC  {len(current_sequence)} frames"
                color = (0, 0, 255)
            else:
                status = "BEREIT  [LEERTASTE]"
                color = (0, 200, 0)

            if not hand_detected:
                cv2.putText(frame, "Keine Hand erkannt", (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 1)

            cv2.putText(frame, f"[{label}] {status}  ({recordings_done}/{times})",
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imshow("Labeling", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if not recording:
                    recording = True
                    current_sequence = []
                    print("  Aufnahme gestartet...")
                else:
                    recording = False
                    n_frames = len(current_sequence)
                    if n_frames >= 10:
                        seq_array = np.array(current_sequence, dtype=np.float32)
                        save_path = data_dir / f"{recording_idx:04d}.pkl"
                        with open(save_path, "wb") as f:
                            pickle.dump(seq_array, f)
                        print(f"  Gespeichert ({n_frames} Frames): {save_path}")
                        recording_idx += 1
                        recordings_done += 1
                    else:
                        print(f"  Zu kurz ({n_frames} Frames), verworfen. Bitte laenger halten.")
                    current_sequence = []

            elif key == ord("q") or key == 27:  # Q oder ESC
                break

        cap.release()
        cv2.destroyAllWindows()

    print(f"Fertig! {recordings_done} Aufnahmen fuer '{label}' gespeichert.")


def dataset_building(output_path):
    """
    Lade alle aufgenommenen Sequenzen, normalisiere sie und speichere
    den fertigen Datensatz für den HMMClassifier.

    Die Normalisierung (z-score pro Sequenz) macht Sequenzen unabhängig
    von der absoluten Handposition und -größe.

    Parameters
    ----------
    output_path : str or Path
        Zielpfad für den erzeugten Datensatz (.pkl).

    Returns
    -------
    sequences : list of np.ndarray
    labels : list of str
    """
    sequences = []
    labels = []

    if not DATA_DIR.exists():
        print(f"Kein Datenordner gefunden: {DATA_DIR}")
        return sequences, labels

    for label_dir in sorted(DATA_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name

        pkl_files = sorted(label_dir.glob("*.pkl"))
        if not pkl_files:
            continue

        for pkl_file in pkl_files:
            with open(pkl_file, "rb") as f:
                seq = pickle.load(f)

            if len(seq) < 5:
                continue

            # z-score Normalisierung pro Sequenz
            mean = seq.mean(axis=0)
            std = seq.std(axis=0) + 1e-8
            seq_norm = (seq - mean) / std

            sequences.append(seq_norm.astype(np.float32))
            labels.append(label)

    print(
        f"Dataset: {len(sequences)} Sequenzen, "
        f"{len(set(labels))} Klassen: {sorted(set(labels))}"
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"sequences": sequences, "labels": labels}, f)
    print(f"Gespeichert: {output_path}")

    return sequences, labels
