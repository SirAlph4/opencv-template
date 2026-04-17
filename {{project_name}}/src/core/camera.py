import sys
import cv2
from src.core.config import DEVICE, BACKEND, FOURCC, WIDTH, HEIGHT, FPS


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(DEVICE, BACKEND)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Could not open camera device {DEVICE}")
    cap.set(cv2.CAP_PROP_FOURCC,       FOURCC)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_GAIN,         0)
    return cap


def verify_caps(cap: cv2.VideoCapture) -> None:
    w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    cc     = int(cap.get(cv2.CAP_PROP_FOURCC))
    cc_str = "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4))
    print(f"[CAP] {w}x{h}  {fps}fps  {cc_str}")
    if (w, h) != (WIDTH, HEIGHT):
        print(f"[WARN] Expected {WIDTH}x{HEIGHT}")


def drain(cap: cv2.VideoCapture, n: int) -> None:
    print(f"[INFO] Draining {n} frames...")
    for _ in range(n):
        cap.read()