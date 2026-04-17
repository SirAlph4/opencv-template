import cv2
import numpy as np
from src.core.config import WIDTH, HEIGHT, HAND_CONNECTIONS


def draw_fps(frame: np.ndarray, fps: int) -> None:
    cv2.putText(frame, f"FPS: {fps}", (40, 50),
                cv2.FONT_HERSHEY_TRIPLEX, 1, (230, 230, 230), 1)


def draw_mode_banner(frame: np.ndarray, mode_name: str) -> None:
    y = frame.shape[0] - 20
    cv2.putText(frame, f"MODE: {mode_name}", (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)


def draw_status(frame: np.ndarray, text: str, y: int = 90) -> None:
    cv2.putText(frame, text, (30, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)


def draw_skeleton(frame: np.ndarray, pixel_pts: np.ndarray) -> None:
    segments = pixel_pts[HAND_CONNECTIONS]
    cv2.polylines(frame, list(segments.reshape(-1, 2, 1, 2)),
                  isClosed=False, color=(0, 255, 0), thickness=2)
    for pt in pixel_pts:
        cv2.circle(frame, tuple(pt), 6, (22, 21, 121), cv2.FILLED)


def draw_hand_bbox(frame: np.ndarray, pixel_pts: np.ndarray,
                   label: str, score: int, padding: int = 50) -> None:
    x_min = max(0,      int(np.min(pixel_pts[:, 0])) - padding)
    y_min = max(0,      int(np.min(pixel_pts[:, 1])) - padding)
    x_max = min(WIDTH,  int(np.max(pixel_pts[:, 0])) + padding)
    y_max = min(HEIGHT, int(np.max(pixel_pts[:, 1])) + padding)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} {score}%", (x_min, y_max + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)