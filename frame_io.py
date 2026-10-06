"""Read and write extracted images through Unicode-safe filesystem APIs."""
from pathlib import Path


def read_frame(path, flags):
    import cv2
    import numpy as np

    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    if not data:
        return None
    return cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)


def write_frame(path, image, params=()):
    import cv2

    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix, image, list(params))
    if not ok:
        return False
    try:
        path.write_bytes(encoded.tobytes())
    except OSError:
        return False
    return True
