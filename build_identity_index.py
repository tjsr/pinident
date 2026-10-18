"""Build a compact, local visual lookup index from stored catalog artwork.

The default output is ignored local data. Use --publish only after every
source's reuse rights have been verified; that writes the web build asset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pin_catalog import DEFAULT_CATALOG, PinCatalog

ROOT = Path(__file__).resolve().parent
LOCAL_OUTPUT = ROOT / 'mobile_web' / 'identity-index.local.json'
PUBLISH_OUTPUT = ROOT / 'mobile_web' / 'identity-index.json'
GRID = 8


def reference_feature(image_bytes: bytes) -> list[float] | None:
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8),
                         cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] not in (3, 4):
        return None
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(np.float64) / 255
        bgr = image[:, :, :3].astype(np.float64) * alpha + 255 * (1 - alpha)
    else:
        bgr = image.astype(np.float64)
    height, width = bgr.shape[:2]
    sampled = np.empty((GRID, GRID, 3), dtype=np.float64)
    for gy in range(GRID):
        py = min(height - 1, int((gy + 0.5) * height / GRID))
        for gx in range(GRID):
            px = min(width - 1, int((gx + 0.5) * width / GRID))
            sampled[gy, gx] = bgr[py, px, ::-1] / 255
    luminance = (sampled[:, :, 0] * 0.299 + sampled[:, :, 1] * 0.587 +
                 sampled[:, :, 2] * 0.114).ravel()
    centred = luminance - luminance.mean()
    norm = np.linalg.norm(centred)
    if norm < 0.18:
        return None
    values = np.concatenate((centred / norm, sampled.mean(axis=(0, 1))))
    return [round(float(value), 7) for value in values]


def build_index(catalog_path: Path) -> dict:
    catalog = PinCatalog(catalog_path)
    references = []
    unverified = 0
    for stock in catalog.stock_training_images():
        feature = reference_feature(stock.image_bytes)
        if feature is None:
            continue
        references.append({
            'pin_id': stock.pin_id,
            'name': stock.tags[0] if stock.tags else stock.name,
            'source': stock.source,
            'rights': stock.rights,
            'feature': feature,
        })
        unverified += 'unverified' in stock.rights.casefold()
    if not references:
        raise ValueError('No usable catalog reference images')
    return {
        'format': 'pinident-catalog-reference-index',
        'version': 1,
        'catalog_state': catalog.stock_training_state(),
        'catalog_rights_unverified': unverified,
        'references': references,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog', type=Path, default=DEFAULT_CATALOG)
    parser.add_argument('--publish', action='store_true',
                        help='write the web asset only when rights are verified')
    args = parser.parse_args()
    index = build_index(args.catalog)
    if args.publish and index['catalog_rights_unverified']:
        raise SystemExit(
            f"Publication held: {index['catalog_rights_unverified']} catalog images "
            'have unverified rights')
    output = PUBLISH_OUTPUT if args.publish else LOCAL_OUTPUT
    output.write_text(json.dumps(index, separators=(',', ':')), encoding='utf-8')
    print(f"Indexed {len(index['references'])} catalog images to {output}")
    print(f"Unverified source rights: {index['catalog_rights_unverified']}")


if __name__ == '__main__':
    main()
