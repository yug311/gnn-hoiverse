"""
vg_image_data.py
----------------
Loads vg/image_data.json and provides fast lookups by image_id.

Usage
-----
    from vg_image_data import get_image_info, get_image_size, get_image_url

    info = get_image_info(1)       # {'image_id': 1, 'width': 800, 'height': 600, 'url': '...'}
    w, h = get_image_size(1)       # (800, 600)
    url  = get_image_url(1)        # 'https://...'
"""

import json
import os

IMAGE_DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "image_data.json")

_image_data: dict | None = None


def _load() -> dict:
    global _image_data
    if _image_data is None:
        with open(IMAGE_DATA_FILE, "r") as f:
            raw = json.load(f)
        _image_data = {entry["image_id"]: entry for entry in raw}
    return _image_data


def get_image_info(image_id: int) -> dict | None:
    """Return the full metadata dict for *image_id*, or None if not found."""
    return _load().get(int(image_id))


def get_image_size(image_id: int) -> tuple[int, int]:
    """Return (width, height) for *image_id*, or (640, 480) as fallback."""
    info = get_image_info(image_id)
    if info is None:
        return 640, 480
    return int(info["width"]), int(info["height"])


def get_image_url(image_id: int) -> str | None:
    """Return the URL string for *image_id*, or None if not found."""
    info = get_image_info(image_id)
    return info["url"] if info else None
