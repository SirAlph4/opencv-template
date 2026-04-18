import re
import numpy as np
from pathlib import Path


def euclidean_distance(pt1: tuple, pt2: tuple) -> float:
  return float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))


def next_session_dir(base_dir: str = "image_data") -> Path:
  root = Path.cwd() / base_dir
  root.mkdir(parents=True, exist_ok=True)

  pattern = re.compile(r"^session_(\d{3})$")
  existing_nums = []

  for p in root.iterdir():
    if p.is_dir():
      m = pattern.match(p.name)
      if m:
        existing_nums.append(int(m.group(1)))

  next_num = max(existing_nums) + 1 if existing_nums else 1
  session_name = f"session_{next_num:03d}"
  session_path = root / session_name
  session_path.mkdir(parents=True, exist_ok=False)
  return session_path
