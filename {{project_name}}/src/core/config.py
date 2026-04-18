# SignLanguageCV — SirAlph4
import os
import cv2
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENCV_VIDEOIO_V4L_RANGE_NORMALIZED"] = "0"

CWD     = Path.cwd()
DEVICE  = int(os.getenv("CAMERA_DEVICE", 0))
BACKEND = cv2.CAP_V4L2
WIDTH   = int(os.getenv("CAMERA_WIDTH",  1920))
HEIGHT  = int(os.getenv("CAMERA_HEIGHT", 1080))
FPS     = 30
FOURCC  = cv2.VideoWriter.fourcc(*"MJPG")
DRAIN_COUNT = 10

DETECTION_CONFIDENCE = 0.8
PRESENCE_CONFIDENCE  = 0.8
TRACKING_CONFIDENCE  = 0.8
MAX_HANDS            = 5
SCORE_THRESHOLD      = 80
IMAGES_PER_CLASS     = 100
COLLECTION_DELAY     = 90
QUIT_GESTURE_DIST    = 1

HAND_CONNECTIONS = np.array([
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [5,9],[9,10],[10,11],[11,12],
    [9,13],[13,14],[14,15],[15,16],
    [13,17],[0,17],[17,18],[18,19],[19,20],
], dtype=np.int32)

CLASSES: dict[int, str] = {
    0:"ZERO", 1:"ONE",  2:"TWO",   3:"THREE", 4:"FOUR",
    5:"FIVE", 6:"SIX",  7:"SEVEN", 8:"EIGHT", 9:"NINE", 10:"TEN",
}