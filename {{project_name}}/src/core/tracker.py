import cv2
import mediapipe as mp
import numpy as np
from src.core.config import (
    WIDTH, HEIGHT,
    DETECTION_CONFIDENCE, PRESENCE_CONFIDENCE,
    TRACKING_CONFIDENCE, MAX_HANDS,
)

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode


class HandState:
    result = None


def _result_callback(result, _frame, _ts) -> None:
    HandState.result = result


def build_handlandmarker(model_path: str = "hand_landmarker.task"):
    from mediapipe.tasks.python import vision
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
        num_hands=MAX_HANDS,
        result_callback=_result_callback,
    )
    return HandLandmarker.create_from_options(options)


def frame_to_mp_image(frame: np.ndarray) -> mp.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(mp.ImageFormat.SRGB, data=rgb)


def landmarks_to_pixels(hand_landmarks: list) -> np.ndarray:
    raw = np.array([(lm.x, lm.y) for lm in hand_landmarks], dtype=np.float32)
    return (raw * np.array([WIDTH, HEIGHT])).astype(np.int32)


def get_fingertips(pixel_pts: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(pixel_pts[i]) for i in (4, 8, 12, 16, 20)]