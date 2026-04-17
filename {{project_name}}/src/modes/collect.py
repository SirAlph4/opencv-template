import cv2
import numpy as np
from src.core.config import (
    CLASSES, WIDTH, HEIGHT,
    IMAGES_PER_CLASS, COLLECTION_DELAY,
    SCORE_THRESHOLD, QUIT_GESTURE_DIST,
)
from src.core.tracker import HandState, landmarks_to_pixels, get_fingertips
from src.utils.drawing import draw_skeleton, draw_hand_bbox, draw_status
from src.utils.fs import euclidean_distance, smart_rotate_folder


class CollectMode:
    NAME = "COLLECT  [SPACE = next class]"

    def __init__(self) -> None:
        self.data_dir   = smart_rotate_folder("data")
        self._make_class_dirs()
        self.class_idx  = 0
        self.counter    = 0
        self.image_cnt  = 0
        self.quit_requested = False

    def _make_class_dirs(self) -> None:
        for name in CLASSES.values():
            for hand in ("Left", "Right"):
                (self.data_dir / name / hand).mkdir(parents=True, exist_ok=True)

    @property
    def current_class(self) -> str:
        return CLASSES[self.class_idx]

    def process(self, frame: np.ndarray) -> None:
        self.counter += 1
        if HandState.result is None or not HandState.result.hand_landmarks:
            self._draw_hud(frame)
            return
        for i, hand_landmarks in enumerate(HandState.result.hand_landmarks):
            pixel_pts = landmarks_to_pixels(hand_landmarks)
            hand_info = HandState.result.handedness[i][0]
            label     = hand_info.category_name
            score     = int(hand_info.score * 100)
            draw_skeleton(frame, pixel_pts)
            draw_hand_bbox(frame, pixel_pts, label, score)
            if (score >= SCORE_THRESHOLD
                    and self.counter >= COLLECTION_DELAY
                    and self.image_cnt < IMAGES_PER_CLASS):
                self._save_crop(frame, pixel_pts, label)
            tips = get_fingertips(pixel_pts)
            if euclidean_distance(tips[0], tips[4]) < QUIT_GESTURE_DIST:
                self.quit_requested = True
        self._draw_hud(frame)

    def _save_crop(self, frame: np.ndarray,
                   pixel_pts: np.ndarray, label: str) -> None:
        pad   = 50
        x_min = max(0,      int(np.min(pixel_pts[:, 0])) - pad)
        y_min = max(0,      int(np.min(pixel_pts[:, 1])) - pad)
        x_max = min(WIDTH,  int(np.max(pixel_pts[:, 0])) + pad)
        y_max = min(HEIGHT, int(np.max(pixel_pts[:, 1])) + pad)
        crop  = frame[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            return
        path = (self.data_dir / self.current_class
                / label / f"{self.image_cnt:04d}.jpg")
        cv2.imwrite(str(path), crop)
        self.image_cnt += 1

    def _draw_hud(self, frame: np.ndarray) -> None:
        remaining = COLLECTION_DELAY - self.counter
        if remaining > 0:
            text = f"'{self.current_class}' starts in {remaining // 30 + 1}s"
        elif self.image_cnt < IMAGES_PER_CLASS:
            text = f"Recording '{self.current_class}': {self.image_cnt}/{IMAGES_PER_CLASS}"
        else:
            text = f"Done! SPACE for next  ({self.class_idx + 1}/{len(CLASSES)})"
        draw_status(frame, text)

    def next_class(self) -> bool:
        self.class_idx += 1
        self.counter    = 0
        self.image_cnt  = 0
        if self.class_idx >= len(CLASSES):
            print("[COLLECT] All classes done.")
            return False
        print(f"[COLLECT] Next: {self.current_class}")
        return True