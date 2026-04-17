import pickle
from pathlib import Path
import cv2
import numpy as np
from src.core.tracker import HandState, landmarks_to_pixels
from src.utils.drawing import draw_skeleton, draw_hand_bbox, draw_status


class InferMode:
    NAME = "INFER"

    def __init__(self, model_path: str = "model.pkl") -> None:
        self._model = None
        if Path(model_path).exists():
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            print(f"[INFER] Model loaded from '{model_path}'.")
        else:
            print(f"[INFER] No model at '{model_path}' — train first.")

    def process(self, frame: np.ndarray) -> None:
        if self._model is None:
            draw_status(frame, "No model. Run Mode 2 (Training) first.")
            return
        if HandState.result is None or not HandState.result.hand_landmarks:
            return
        for i, hand_landmarks in enumerate(HandState.result.hand_landmarks):
            pixel_pts  = landmarks_to_pixels(hand_landmarks)
            hand_info  = HandState.result.handedness[i][0]
            prediction = self._model.predict(
                pixel_pts.flatten().reshape(1, -1))[0]
            draw_skeleton(frame, pixel_pts)
            draw_hand_bbox(frame, pixel_pts,
                           hand_info.category_name,
                           int(hand_info.score * 100))
            cv2.putText(frame, str(prediction), (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 80), 6)