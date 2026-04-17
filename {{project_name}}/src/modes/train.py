import threading
import numpy as np
from src.utils.drawing import draw_status


class TrainMode:
    NAME = "TRAIN  [T = start training]"

    def __init__(self) -> None:
        self._status = "Press T to start training."
        self._thread: threading.Thread | None = None

    def process(self, frame: np.ndarray) -> None:
        draw_status(frame, self._status)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._status = "Loading dataset..."
            # X, y = load_landmarks("data/")
            self._status = "Fitting model..."
            # clf.fit(X_train, y_train)
            self._status = "Saving model..."
            # pickle.dump(clf, open("model.pkl", "wb"))
            self._status = "Done! Switch to Mode 3 for inference."
        except Exception as exc:
            self._status = f"[ERROR] {exc}"