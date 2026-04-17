import time
import cv2
from src.core.camera   import open_camera, verify_caps, drain
from src.core.config   import DRAIN_COUNT
from src.utils.drawing import draw_fps, draw_mode_banner
from src.modes.collect import CollectMode
from src.modes.train   import TrainMode
from src.modes.infer   import InferMode

_MODE_KEYS = {ord("1"): 0, ord("2"): 1, ord("3"): 2}


def main() -> None:
    cap    = open_camera()
    verify_caps(cap)
    drain(cap, DRAIN_COUNT)
    modes  = [CollectMode(), TrainMode(), InferMode()]
    active = 0
    pTime  = time.time()

    print("[KEYS] 1=Collect  2=Train  3=Infer  q=Quit  s=Snapshot")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        cTime = time.time()
        fps   = int(1 / max(cTime - pTime, 1e-9))
        pTime = cTime

        modes[active].process(frame)
        draw_fps(frame, fps)
        draw_mode_banner(frame, modes[active].NAME)
        cv2.imshow("{{ project_name }}  |  1/2/3=mode  q=quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in _MODE_KEYS:
            active = _MODE_KEYS[key]
            print(f"[MODE] {modes[active].NAME}")
        elif key == ord("s"):
            import datetime
            fname = f"snapshot_{datetime.datetime.now():%Y%m%d_%H%M%S}.png"
            cv2.imwrite(fname, frame)
            print(f"[SNAP] {fname}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released.")


if __name__ == "__main__":
    main()