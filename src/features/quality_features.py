"""Image/scan quality features.

blur_score      -- variance of edge response (PIL FIND_EDGES).
                   Higher = sharper. Near-zero = blurry / fax scan.
image_contrast  -- pixel std of grayscale normalised to 0-1 (std / 255).
                   Low (~0.05-0.15) = faded/grey thermal receipt; strong
                   routing signal — these images are hard for cheap OCR.
skew_angle      -- estimated text skew in degrees (-180..180) via horizontal
                   projection profiles at sampled angles. 0 = straight.
estimated_dpi   -- inferred from image height assuming ~180 mm receipt height.
                   Rough proxy; actual DPI is not embedded in the images.
is_grayscale    -- True when the image is already single-channel (L/LA/1).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from src.data.base_loader import Document

_RECEIPT_HEIGHT_MM = 180.0
_MM_PER_INCH = 25.4

_EMPTY = {
    "blur_score"    : 0.0,
    "image_contrast": 0.0,
    "skew_angle"    : 0.0,
    "estimated_dpi" : 0.0,
    "is_grayscale"  : None,
}


def _estimate_skew(gray: Image.Image) -> float:
    """Estimate text skew via horizontal projection profiles.

    Resizes to a small thumbnail for speed, binarises, then tries angles
    from -10° to +10° in 1° steps. The angle whose row-sum profile has
    maximum variance aligns text lines horizontally — that is the skew.
    Returns 0.0 on images with no discernible text structure.
    """
    # Thumbnail: keep aspect ratio, cap width at 300px for speed
    w, h = gray.size
    scale = min(1.0, 300 / w)
    thumb = gray.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    arr = np.asarray(thumb, dtype=np.float32)
    binary = (arr < 127).astype(np.float32)  # dark pixels = text

    best_angle, best_score = 0.0, -np.inf
    for angle in range(-10, 11):
        if angle == 0:
            rotated = binary
        else:
            rotated = np.asarray(
                Image.fromarray((binary * 255).astype(np.uint8)).rotate(
                    angle, expand=False, fillcolor=0
                ),
                dtype=np.float32,
            )
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def extract_quality_features(document: Document) -> dict:
    """Compute image quality signals for one document.

    Returns ``_EMPTY`` values when the image is unavailable.
    """
    if document.image_path is None or not Path(document.image_path).exists():
        return _EMPTY.copy()

    with Image.open(document.image_path) as im:
        is_grayscale = im.mode in ("L", "LA", "1")
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

        edges = gray.filter(ImageFilter.FIND_EDGES)
        blur_score = float(np.var(np.asarray(edges, dtype=np.float32)))

        image_contrast = round(float(arr.std() / 255.0), 4)

        skew_angle = _estimate_skew(gray)

        est_dpi = (im.height / _RECEIPT_HEIGHT_MM) * _MM_PER_INCH

    return {
        "blur_score"    : round(blur_score, 1),
        "image_contrast": image_contrast,
        "skew_angle"    : skew_angle,
        "estimated_dpi" : round(est_dpi, 1),
        "is_grayscale"  : is_grayscale,
    }
