import re
import numpy as np
from pathlib import Path


def euclidean_distance(pt1: tuple, pt2: tuple) -> float:
    return float(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))


def smart_rotate_folder(base_name: str = "data", cwd: Path = Path.cwd()) -> Path:
    target = cwd / base_name
    if not target.exists():
        target.mkdir(parents=True)
        print(f"[FS] Created: {base_name}/")
        return target
    pattern = re.compile(rf"^{re.escape(base_name)}_(\d+)$")
    existing = [int(m.group(1)) for f in cwd.glob(f"{base_name}_*")
                if (m := pattern.match(f.name))]
    next_num = max(existing) + 1 if existing else 2
    target.rename(cwd / f"{base_name}_{next_num}")
    target.mkdir()
    print(f"[FS] Rotated → {base_name}_{next_num}/  Created fresh {base_name}/")
    return target