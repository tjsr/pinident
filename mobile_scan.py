"""Frame scanning shared by the mobile HTTP entry point and focused tests.

The service proposes possible pins and runs the existing catalog matcher. Results
are suggestions until a person tags or adds a pin to their library.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boxdata import BoxData
from pin_catalog import PinMatcher
from pin_detection import associate_frame_boxes, detect_pin_boxes, identify_candidates


@dataclass
class ScanState:
    image: np.ndarray
    boxes: list[BoxData]


def scan_frame(image: np.ndarray, matcher: PinMatcher,
               previous: ScanState | None = None) -> ScanState:
    """Detect a frame; keep identities and track IDs only for visual matches."""
    proposals = detect_pin_boxes(image)
    if previous is None:
        boxes = [BoxData(coords, [], 'automatic') for coords in proposals]
    else:
        boxes = associate_frame_boxes(previous.image, image, previous.boxes, proposals)
    # A stable match carried from the previous frame is already catalogued.
    identify_candidates(image, [box for box in boxes if box.pin_id is None], matcher)
    return ScanState(image, boxes)


def public_scan_result(state: ScanState) -> dict:
    height, width = state.image.shape[:2]
    return {
        'width': width,
        'height': height,
        'boxes': [
            {'coords': list(box.coords), 'track_id': box.track_id,
             'pin_id': box.pin_id,
             'review_state': box.review_state,
             'name': next((tag for tag in box.tags or [] if tag.strip()), None)}
            for box in state.boxes if box.review_state != 'rejected'
        ],
    }
