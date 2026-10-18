"""Two-stage, class-agnostic pin proposals and conservative frame association.

All coordinates are in the source RGB image, as (x, y, width, height).
These heuristics find *possible* pins; they do not identify a Pinny Arcade design.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from boxdata import (BoxData, Coordinate, has_specific_identity,
                     rejects_identity, same_specific_name)
from detection_feedback import FeedbackModel
from pin_catalog import PinMatcher


@dataclass(frozen=True)
class DetectionSettings:
    max_working_side: int = 960
    min_side_fraction: float = 0.04
    max_side_fraction: float = 0.32
    max_candidates: int = 150


def _validate_image(image: np.ndarray) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an RGB uint8 image with three channels")


def _intersection_over_union(a: Coordinate, b: Coordinate) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0.0


def _intersection_area(a: Coordinate, b: Coordinate) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (max(0, min(ax + aw, bx + bw) - max(ax, bx)) *
            max(0, min(ay + ah, by + bh) - max(ay, by)))


def _strongly_overlaps(a: Coordinate, b: Coordinate) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if min(aw, ah, bw, bh) <= 0:
        return False
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return _intersection_over_union(a, b) >= 0.35 or intersection / min(aw * ah, bw * bh) > 0.50


def _has_local_contrast(image: np.ndarray, region: Coordinate,
                        threshold: float = 15.0) -> bool:
    x, y, w, h = region
    pad = max(4, int(0.25 * max(w, h)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(image.shape[1], x + w + pad), min(image.shape[0], y + h + pad)
    lab = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_RGB2LAB)
    inner = lab[y - y0 + int(0.2 * h):y - y0 + int(0.8 * h),
                x - x0 + int(0.2 * w):x - x0 + int(0.8 * w)]
    if inner.size == 0:
        return False
    difference = np.linalg.norm(np.median(inner.reshape(-1, 3), axis=0) -
                                np.median(lab.reshape(-1, 3), axis=0))
    if difference >= threshold:
        return True
    # A thin fragment at the video boundary can dominate its own padded
    # neighborhood, making the median-to-median contrast look deceptively low.
    exterior = None
    if x + w >= image.shape[1]:
        exterior = image[y:y + h, max(0, x - max(4, w)):x]
    elif x <= 0:
        exterior = image[y:y + h, x + w:min(image.shape[1], x + 2 * w)]
    elif y + h >= image.shape[0]:
        exterior = image[max(0, y - max(4, h)):y, x:x + w]
    elif y <= 0:
        exterior = image[y + h:min(image.shape[0], y + 2 * h), x:x + w]
    if exterior is None or exterior.size == 0:
        return False
    fragment = image[y:y + h, x:x + w]
    if float(cv2.cvtColor(fragment, cv2.COLOR_RGB2GRAY).std()) < 12.0:
        return False
    background = np.median(exterior.reshape(-1, 3), axis=0)
    distances = np.linalg.norm(fragment.astype(np.float32) - background, axis=2)
    return float(np.percentile(distances, 70)) >= 45.0


def propose_regions(image: np.ndarray, settings: DetectionSettings = DetectionSettings()) -> list[Coordinate]:
    """Find coarse, bounded high-contrast regions at a reduced resolution."""
    _validate_image(image)
    height, width = image.shape[:2]
    if min(height, width) < 24:
        return []

    scale = min(1.0, settings.max_working_side / max(height, width))
    working = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    median = float(np.median(gray))
    edges = cv2.Canny(gray, max(18, int(0.55 * median)), max(50, int(1.4 * median)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    working_height, working_width = working.shape[:2]
    min_side = max(10, int(min(working_height, working_width) * settings.min_side_fraction))
    max_side = max(1, int(max(working_height, working_width) * settings.max_side_fraction))
    proposals: list[Coordinate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min(w, h) < min_side or max(w, h) > max_side:
            continue
        clipped = (x <= 1 or y <= 1 or x + w >= working_width - 1 or
                   y + h >= working_height - 1)
        if (max(w, h) / min(w, h) > (6.0 if clipped else 3.0) or
                cv2.contourArea(contour) < min_side * min_side * 0.3):
            continue
        pad = max(4, int(0.12 * max(w, h)))
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(working_width, x + w + pad)
        bottom = min(working_height, y + h + pad)
        proposals.append((
            int(left / scale), int(top / scale),
            min(width, int(np.ceil(right / scale))) - int(left / scale),
            min(height, int(np.ceil(bottom / scale))) - int(top / scale),
        ))

    # A second coarse cue catches colorful silhouettes whose outlines touch
    # printed artwork or the edge of a backing card.
    lab = cv2.cvtColor(working, cv2.COLOR_RGB2LAB).astype(np.float32)
    background = cv2.GaussianBlur(lab, (0, 0), 24)
    difference = np.linalg.norm(lab - background, axis=2)
    color_mask = np.uint8(difference > 40) * 255
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    count, _, components, _ = cv2.connectedComponentsWithStats(color_mask)
    for index in range(1, count):
        x, y, w, h, area = components[index]
        clipped = (x <= 1 or y <= 1 or x + w >= working_width - 1 or
                   y + h >= working_height - 1)
        if (min(w, h) < min_side or max(w, h) > max_side or
                max(w, h) / min(w, h) > (6.0 if clipped else 3.0)):
            continue
        if area < min_side * min_side * 0.3:
            continue
        pad = max(4, int(0.10 * max(w, h)))
        left, top = max(0, x - pad), max(0, y - pad)
        right, bottom = min(working_width, x + w + pad), min(working_height, y + h + pad)
        proposals.append((
            int(left / scale), int(top / scale),
            min(width, int(np.ceil(right / scale))) - int(left / scale),
            min(height, int(np.ceil(bottom / scale))) - int(top / scale),
        ))

    # Larger contours tend to describe the complete object rather than its details.
    proposals.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
    return [rect for rect in proposals[:settings.max_candidates] if _has_local_contrast(image, rect)]


def refine_region(image: np.ndarray, region: Coordinate) -> Coordinate | None:
    """Find the main foreground component inside a coarse region.

    The outside rim estimates local background color. If it cannot isolate a
    component, the coarse region is retained by the caller for human review.
    """
    _validate_image(image)
    height, width = image.shape[:2]
    x, y, w, h = region
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(width, x + w), min(height, y + h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    crop = image[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
    rim = max(2, int(min(crop.shape[:2]) * 0.12))
    border = np.concatenate((lab[:rim].reshape(-1, 3), lab[-rim:].reshape(-1, 3),
                             lab[:, :rim].reshape(-1, 3), lab[:, -rim:].reshape(-1, 3)))
    background = np.median(border, axis=0)
    distance = np.linalg.norm(lab - background, axis=2)
    border_distance = np.linalg.norm(border - background, axis=1)
    threshold = max(20.0, float(np.percentile(border_distance, 75)) + 10.0)
    mask = np.uint8(distance > threshold) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return None

    center = np.array([crop.shape[1] / 2, crop.shape[0] / 2])
    candidates = []
    for index in range(1, count):
        bx, by, bw, bh, area = stats[index]
        if area < max(20, 0.01 * crop.shape[0] * crop.shape[1]):
            continue
        distance_from_center = np.linalg.norm((centroids[index] - center) / np.array([crop.shape[1], crop.shape[0]]))
        candidates.append((area / (1 + 4 * distance_from_center), (x0 + bx, y0 + by, bw, bh)))
    if not candidates:
        return None
    bx, by, bw, bh = max(candidates, key=lambda candidate: candidate[0])[1]
    return int(bx), int(by), int(bw), int(bh)


def detect_pin_boxes(image: np.ndarray, settings: DetectionSettings = DetectionSettings(),
                     feedback_model: FeedbackModel | None = None) -> list[Coordinate]:
    """Propose regions, refine locally, then suppress overlapping results."""
    boxes = []
    for region in propose_regions(image, settings):
        refined = refine_region(image, region)
        # A small fragment is usually an interior color patch, not the pin edge.
        boxes.append(refined if refined and refined[2] * refined[3] >= 0.4 * region[2] * region[3] else region)
    boxes.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
    if feedback_model is not None:
        boxes = feedback_model.filter_and_rank(image, boxes)
    result: list[Coordinate] = []
    for box in boxes:
        if all(not _strongly_overlaps(box, accepted) for accepted in result):
            result.append(box)
    return result[:60]


def detect_pin_boxes_relaxed(image: np.ndarray,
                             feedback_model: FeedbackModel | None = None) -> list[Coordinate]:
    """Additional contrast-normalized contour search for user-asserted misses.

    This search is deliberately only used after ordinary detection misses the
    number of pins the user says are present; it can produce more false positives.
    """
    _validate_image(image)
    height, width = image.shape[:2]
    if min(height, width) < 24:
        return []
    scale = min(1.0, 960 / max(height, width))
    working = (cv2.resize(image, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA) if scale < 1 else image)
    gray = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
    equalized = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(equalized, 25, 75)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    small_side = min(working.shape[:2])
    min_side = max(8, int(0.025 * small_side))
    max_side = max(1, int(0.40 * max(working.shape[:2])))
    regions: list[Coordinate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        clipped = (x <= 1 or y <= 1 or x + w >= working.shape[1] - 1 or
                   y + h >= working.shape[0] - 1)
        contour_area = cv2.contourArea(contour)
        if (min(w, h) < min_side or max(w, h) > max_side or
                max(w, h) / min(w, h) > 2.5 or
                contour_area < 0.25 * min_side * min_side or
                (not clipped and contour_area < 0.48 * w * h)):
            continue
        if float(np.std(equalized[y:y+h, x:x+w])) < 5:
            continue
        pad = max(3, int(0.08 * max(w, h)))
        left, top = max(0, x-pad), max(0, y-pad)
        right = min(working.shape[1], x+w+pad)
        bottom = min(working.shape[0], y+h+pad)
        sx, sy = int(left / scale), int(top / scale)
        regions.append((sx, sy, min(width, int(np.ceil(right / scale))) - sx,
                        min(height, int(np.ceil(bottom / scale))) - sy))
    regions.sort(key=lambda rect: rect[2] * rect[3], reverse=True)
    # CLAHE exaggerates repeating surface texture (wood grain, fabric) into
    # many tiny closed contours. Keep the weaker genuine silhouettes while
    # requiring a visible difference from their surrounding image pixels.
    regions = [region for region in regions[:200]
               if _has_local_contrast(image, region, threshold=12.0)]
    if feedback_model is not None:
        regions = feedback_model.filter_and_rank(image, regions)
    accepted: list[Coordinate] = []
    for region in regions:
        if all(not _strongly_overlaps(region, old) for old in accepted):
            accepted.append(region)
        if len(accepted) >= 40:
            break
    return accepted


def _appearance_similarity(previous: np.ndarray, current: np.ndarray, a: Coordinate, b: Coordinate) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    first = previous[ay:ay + ah, ax:ax + aw]
    second = current[by:by + bh, bx:bx + bw]
    if first.size == 0 or second.size == 0:
        return 0.0
    first_lab = cv2.cvtColor(first, cv2.COLOR_RGB2LAB)
    second_lab = cv2.cvtColor(second, cv2.COLOR_RGB2LAB)
    first_chroma = np.median(first_lab.reshape(-1, 3), axis=0)[1:]
    second_chroma = np.median(second_lab.reshape(-1, 3), axis=0)[1:]
    if np.linalg.norm(first_chroma - second_chroma) > 30:
        return 0.0
    first = cv2.resize(cv2.cvtColor(first, cv2.COLOR_RGB2GRAY), (32, 32)).astype(np.float32)
    second = cv2.resize(cv2.cvtColor(second, cv2.COLOR_RGB2GRAY), (32, 32)).astype(np.float32)
    first -= first.mean()
    second -= second.mean()
    norm = float(np.linalg.norm(first) * np.linalg.norm(second))
    return max(0.0, float(np.sum(first * second) / norm)) if norm > 1e-6 else 0.0


def _near_frame_edge(image: np.ndarray, coords: Coordinate) -> bool:
    x, y, width, height = coords
    margin = max(8, int(0.25 * max(width, height)))
    return (x <= margin or y <= margin or
            image.shape[1] - x - width <= margin or
            image.shape[0] - y - height <= margin)


def _edge_expansion_similarity(first_frame: np.ndarray, second_frame: np.ndarray,
                               first_box: Coordinate, second_box: Coordinate) -> float:
    """Compare a clipped edge fragment with part of a larger adjacent box."""
    first_area = first_box[2] * first_box[3]
    second_area = second_box[2] * second_box[3]
    if first_area <= 0 or second_area <= 0:
        return 0.0
    if first_area <= 0.8 * second_area:
        small_frame, small_box = first_frame, first_box
        large_frame, large_box = second_frame, second_box
    elif second_area <= 0.8 * first_area:
        small_frame, small_box = second_frame, second_box
        large_frame, large_box = first_frame, first_box
    else:
        return 0.0
    if not _near_frame_edge(small_frame, small_box):
        return 0.0
    sx, sy, sw, sh = small_box
    lx, ly, lw, lh = large_box
    if (sx < 0 or sy < 0 or lx < 0 or ly < 0 or
            sx + sw > small_frame.shape[1] or sy + sh > small_frame.shape[0] or
            lx + lw > large_frame.shape[1] or ly + lh > large_frame.shape[0] or
            sw > lw or sh > lh):
        return 0.0
    if (abs((sx + sw / 2) - (lx + lw / 2)) > 1.5 * lw or
            abs((sy + sh / 2) - (ly + lh / 2)) > 1.5 * lh):
        return 0.0
    template = cv2.cvtColor(small_frame[sy:sy + sh, sx:sx + sw], cv2.COLOR_RGB2GRAY)
    if float(template.std()) < 8.0:
        return 0.0
    search = cv2.cvtColor(large_frame[ly:ly + lh, lx:lx + lw], cv2.COLOR_RGB2GRAY)
    _, score, _, location = cv2.minMaxLoc(
        cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED))
    if score < 0.82:
        return 0.0
    aligned = (lx + location[0], ly + location[1], sw, sh)
    appearance = _appearance_similarity(small_frame, large_frame, small_box, aligned)
    return appearance if appearance >= 0.70 else 0.0


def _local_foreground_box(image: np.ndarray, seed: Coordinate,
                          wide: bool = False) -> Coordinate | None:
    """Fit a nearby component around a detector or tracked rectangle."""
    x, y, w, h = seed
    if w <= 0 or h <= 0:
        return None
    pad = round(max(w, h) * (1.5 if wide or _near_frame_edge(image, seed) else 0.75))
    left, top = max(0, x - pad), max(0, y - pad)
    right = min(image.shape[1], x + w + pad)
    bottom = min(image.shape[0], y + h + pad)
    fitted = refine_region(image, (left, top, right - left, bottom - top))
    if fitted is None:
        return None
    overlap = _intersection_area(seed, fitted)
    small_area = min(w * h, fitted[2] * fitted[3])
    limit = 5.0 if wide or _near_frame_edge(image, seed) else 4.0
    if (overlap < 0.65 * small_area or
            fitted[2] * fitted[3] > limit * w * h or
            fitted[2] > limit * w or fitted[3] > limit * h or
            w > limit * fitted[2] or h > limit * fitted[3]):
        return None
    return fitted


def _strong_tracked_fit(previous: np.ndarray, current: np.ndarray,
                        source: Coordinate, fitted: Coordinate | None) -> bool:
    return fitted is not None and (
        _appearance_similarity(previous, current, source, fitted) >= 0.65 or
        _edge_expansion_similarity(previous, current, source, fitted) >= 0.70)


def _search_for_previous_box(previous: np.ndarray, current: np.ndarray, coords: Coordinate) -> Coordinate | None:
    """Search a small neighborhood for an unchanged pin when proposals miss it."""
    x, y, w, h = coords
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > previous.shape[1] or y + h > previous.shape[0]:
        return None
    template = previous[y:y + h, x:x + w]
    if float(np.std(template.astype(np.float32), axis=(0, 1)).mean()) < 8:
        return None
    # Local segmentation can grow across background texture. Only use it when
    # the source rectangle is shaped like a partial object at the frame edge.
    clipped = (_near_frame_edge(previous, coords) and
               min(w, h) <= 0.6 * max(w, h))
    padding = int((1.5 if clipped else 0.75) * max(w, h))
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(current.shape[1], x + w + padding)
    bottom = min(current.shape[0], y + h + padding)
    search = current[top:bottom, left:right]
    if search.shape[0] < h or search.shape[1] < w:
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    candidate = (left + location[0], top + location[1], w, h)
    matched = (score >= 0.82 and
               _appearance_similarity(previous, current, coords, candidate) >= 0.65)
    seed = candidate if matched else coords
    fitted = _local_foreground_box(current, seed, wide=clipped)
    if _strong_tracked_fit(previous, current, coords, fitted):
        return fitted
    return candidate if matched else None


def _carry_user_bounds(previous: np.ndarray, current: np.ndarray,
                       source: BoxData,
                       detection: Coordinate) -> tuple[Coordinate, Coordinate | None]:
    """Keep a user-authored crop while a detector supplies motion/association."""
    adjustment = source.bounds_adjustment
    if adjustment is None:
        return detection, None
    x, y, width, height = detection
    if source.source == 'user' or adjustment == (0, 0, 0, 0):
        located = _search_for_previous_box(previous, current, source.coords)
        if located is not None and _strongly_overlaps(located, detection):
            return located, tuple(located[i] - detection[i] for i in range(4))
    if source.source == 'user':
        # A drawn box has no original detector proposal from which to derive
        # margins. Retain its dimensions around the matched proposal.
        width = min(source.coords[2], current.shape[1])
        height = min(source.coords[3], current.shape[0])
        left = max(0, min(round(x + (detection[2] - width) / 2), current.shape[1] - width))
        top = max(0, min(round(y + (detection[3] - height) / 2), current.shape[0] - height))
        carried = (left, top, width, height)
        return carried, tuple(carried[i] - detection[i] for i in range(4))
    dx, dy, dw, dh = adjustment
    original_width = source.coords[2] - dw
    original_height = source.coords[3] - dh
    scale_x = min(2.0, max(0.5, width / original_width)) if original_width > 0 else 1.0
    scale_y = min(2.0, max(0.5, height / original_height)) if original_height > 0 else 1.0
    dx, dw = round(dx * scale_x), round(dw * scale_x)
    dy, dh = round(dy * scale_y), round(dh * scale_y)
    left = max(0, min(x + dx, current.shape[1] - 1))
    top = max(0, min(y + dy, current.shape[0] - 1))
    right = max(left + 1, min(x + dx + width + dw, current.shape[1]))
    bottom = max(top + 1, min(y + dy + height + dh, current.shape[0]))
    carried = (left, top, right - left, bottom - top)
    return carried, tuple(carried[i] - detection[i] for i in range(4))


def inherit_corrected_bounds(previous: np.ndarray, current: np.ndarray,
                             source: BoxData, target: BoxData) -> bool:
    """Update an untouched detector crop from adjacent tracked geometry."""
    if (target.source != 'automatic' or target.bounds_direct or
            target.feedback_id is not None or
            target.review_state not in ('unconfirmed', 'inherited')):
        return False
    detector = target.coords
    if target.bounds_adjustment is not None:
        detector = tuple(target.coords[i] - target.bounds_adjustment[i]
                         for i in range(4))
        x, y, w, h = detector
        if (x < 0 or y < 0 or w <= 0 or h <= 0 or
                x + w > current.shape[1] or y + h > current.shape[0]):
            detector = target.coords
    if source.bounds_adjustment is None:
        if target.bounds_adjustment is not None:
            if target.track_id != source.track_id:
                return False
            coords, adjustment = detector, None
        else:
            fitted = _local_foreground_box(current, detector)
            if not _strong_tracked_fit(previous, current, source.coords, fitted):
                return False
            coords, adjustment = fitted, None
    else:
        fitted = _local_foreground_box(current, detector)
        if (_strong_tracked_fit(previous, current, source.coords, fitted) and
                (abs(fitted[2] - detector[2]) > 0.1 * detector[2] or
                 abs(fitted[3] - detector[3]) > 0.1 * detector[3])):
            detector = fitted
        coords, adjustment = _carry_user_bounds(previous, current, source, detector)
    if coords == target.coords and adjustment == target.bounds_adjustment:
        return False
    target.coords = coords
    target.bounds_adjustment = adjustment
    return True


def associate_frame_boxes(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    previous_boxes: list[BoxData],
    detections: list[Coordinate],
) -> list[BoxData]:
    """Carry strong visual matches, including hidden rejected-object observations."""
    _validate_image(previous_frame)
    _validate_image(current_frame)
    matches = []
    fitted_detections: dict[int, Coordinate | None] = {}
    for old_index, old_box in enumerate(previous_boxes):
        for new_index, detection in enumerate(detections):
            distance = max(abs(old_box.coords[0] + old_box.coords[2] / 2 -
                               detection[0] - detection[2] / 2),
                           abs(old_box.coords[1] + old_box.coords[3] / 2 -
                               detection[1] - detection[3] / 2))
            candidate = detection
            if distance <= 1.5 * max(*old_box.coords[2:], *detection[2:]):
                if new_index not in fitted_detections:
                    fitted_detections[new_index] = _local_foreground_box(
                        current_frame, detection)
                fitted = fitted_detections[new_index]
                if (_strong_tracked_fit(previous_frame, current_frame,
                                        old_box.coords, fitted) and
                        (abs(fitted[2] - detection[2]) > 0.1 * detection[2] or
                         abs(fitted[3] - detection[3]) > 0.1 * detection[3])):
                    candidate = fitted
            overlap = _intersection_over_union(old_box.coords, candidate)
            appearance = (_appearance_similarity(previous_frame, current_frame,
                                                 old_box.coords, candidate)
                          if overlap >= 0.20 else 0.0)
            edge_match = (0.0 if appearance >= 0.65 else
                          _edge_expansion_similarity(previous_frame, current_frame,
                                                     old_box.coords, candidate))
            if (overlap >= 0.20 and appearance >= 0.65) or edge_match >= 0.70:
                matches.append((old_box.review_state != 'rejected',
                                0.6 * max(overlap, 0.35 if edge_match else 0.0) +
                                0.4 * max(appearance, edge_match), old_index, new_index,
                                candidate))
    matches.sort(reverse=True)
    assigned_old: set[int] = set()
    assigned_new: dict[int, BoxData] = {}
    for _, _, old_index, new_index, candidate in matches:
        if old_index in assigned_old or new_index in assigned_new:
            continue
        old_box = previous_boxes[old_index]
        state = ('rejected' if old_box.review_state == 'rejected' else
                 'inherited' if old_box.review_state != 'unconfirmed' else 'unconfirmed')
        carried_coords, adjustment = _carry_user_bounds(
            previous_frame, current_frame, old_box, candidate)
        tracked = BoxData(carried_coords,
                          (list(old_box.tags or []) if state != 'rejected' and
                           has_specific_identity(old_box) else []),
                          'automatic', old_box.track_id,
                          None if state == 'rejected' else old_box.pin_id, state,
                          match_confidence=(None if state == 'rejected' else
                                            old_box.match_confidence),
                          identity_confirmed=(False if state == 'rejected' else
                                              old_box.identity_confirmed),
                          identity_rejections=old_box.identity_rejections,
                          bounds_adjustment=adjustment)
        tracked.prev_box_id = old_box.id
        assigned_old.add(old_index)
        assigned_new[new_index] = tracked
    result = [assigned_new.get(index) or BoxData(coords, [], 'automatic')
              for index, coords in enumerate(detections)]
    for old_index, old_box in enumerate(previous_boxes):
        if old_index in assigned_old or (not has_specific_identity(old_box)
                                         and old_box.review_state == 'unconfirmed'
                                         and old_box.bounds_adjustment is None
                                         and not _near_frame_edge(previous_frame,
                                                                  old_box.coords)):
            continue
        candidate = _search_for_previous_box(previous_frame, current_frame, old_box.coords)
        if candidate is None:
            continue
        overlaps = [box for box in result if _strongly_overlaps(candidate, box.coords)]
        if any(box.tags or box.source == 'user' or
               box.review_state in ('confirmed', 'inherited') for box in overlaps):
            continue
        result = [box for box in result if box not in overlaps]
        state = ('rejected' if old_box.review_state == 'rejected' else
                 'inherited' if old_box.review_state != 'unconfirmed' else 'unconfirmed')
        tracked = BoxData(candidate,
                          (list(old_box.tags or []) if state != 'rejected' and
                           has_specific_identity(old_box) else []),
                          'automatic', old_box.track_id,
                          None if state == 'rejected' else old_box.pin_id, state,
                          match_confidence=(None if state == 'rejected' else
                                            old_box.match_confidence),
                          identity_confirmed=(False if state == 'rejected' else
                                              old_box.identity_confirmed),
                          identity_rejections=old_box.identity_rejections,
                          bounds_adjustment=((0, 0, 0, 0)
                                             if old_box.bounds_adjustment is not None
                                             else None))
        tracked.prev_box_id = old_box.id
        result.append(tracked)
    return result


def can_link_saved_target(source: BoxData, target: BoxData) -> bool:
    """Keep different manual names separate while accepting the same confirmed name."""
    if target.source not in ('automatic', 'user') or target.review_state == 'rejected':
        return False
    if target.review_state == 'confirmed':
        if source.review_state not in ('confirmed', 'inherited'):
            return False
        if source.identity_confirmed and target.identity_confirmed:
            if source.pin_id and target.pin_id and source.pin_id != target.pin_id:
                return False
            if (not (source.pin_id and target.pin_id)
                    and has_specific_identity(source) and has_specific_identity(target)
                    and not same_specific_name(source, target)):
                return False
        if (target.source == 'user' and has_specific_identity(target)
                and source.identity_confirmed
                and not (source.pin_id and source.pin_id == target.pin_id)
                and not same_specific_name(source, target)):
            return False
        return True
    if target.review_state in ('confirmed', 'inherited') and (
            target.identity_confirmed or not source.identity_confirmed):
        return (target.review_state == 'inherited' and
                target.identity_confirmed and source.identity_confirmed and
                ((bool(source.pin_id) and source.pin_id == target.pin_id) or
                 (not source.pin_id and not target.pin_id and
                  same_specific_name(source, target))))
    if target.source == 'user':
        if not has_specific_identity(target):
            return source.review_state in ('confirmed', 'inherited')
        return source.identity_confirmed and same_specific_name(source, target)
    if target.pin_id:
        return True  # Automatic catalog guesses can be corrected by a verified source.
    if has_specific_identity(target):
        return source.identity_confirmed and same_specific_name(source, target)
    return True


def match_rejected_candidates(
    reference_frame: np.ndarray,
    current_frame: np.ndarray,
    rejected_boxes: list[BoxData],
    current_boxes: list[BoxData],
) -> list[tuple[BoxData, BoxData]]:
    """Find saved automatic candidates that are the same rejected object."""
    _validate_image(reference_frame)
    _validate_image(current_frame)
    matches: list[tuple[float, int, int]] = []
    for source_index, source in enumerate(rejected_boxes):
        if source.review_state != 'rejected':
            continue
        for target_index, target in enumerate(current_boxes):
            if target.source != 'automatic' or target.review_state != 'unconfirmed':
                continue
            overlap = _intersection_over_union(source.coords, target.coords)
            if overlap < 0.20:
                continue
            appearance = _appearance_similarity(reference_frame, current_frame,
                                                source.coords, target.coords)
            if appearance >= 0.75:
                matches.append((0.6 * overlap + 0.4 * appearance,
                                source_index, target_index))
    matches.sort(reverse=True)
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    result: list[tuple[BoxData, BoxData]] = []
    for _, source_index, target_index in matches:
        if source_index not in used_sources and target_index not in used_targets:
            result.append((rejected_boxes[source_index], current_boxes[target_index]))
            used_sources.add(source_index)
            used_targets.add(target_index)
    return result


def match_existing_candidates(
    reference_frame: np.ndarray,
    current_frame: np.ndarray,
    reference_boxes: list[BoxData],
    current_boxes: list[BoxData],
) -> list[tuple[BoxData, BoxData]]:
    """Match adjacent observations using position, appearance and compatible labels."""
    _validate_image(reference_frame)
    _validate_image(current_frame)
    matches: list[tuple[float, int, int]] = []
    fitted_targets: dict[int, Coordinate | None] = {}
    for source_index, source in enumerate(reference_boxes):
        has_correction_link = (source.bounds_adjustment is not None or any(
            target.track_id == source.track_id and
            target.bounds_adjustment is not None and not target.bounds_direct
            for target in current_boxes))
        if source.review_state == 'rejected' or not (
                source.review_state in ('confirmed', 'inherited') or
                has_specific_identity(source) or
                has_correction_link or
                (source.review_state == 'unconfirmed' and
                 source.source == 'automatic')):
            continue
        for target_index, target in enumerate(current_boxes):
            if not can_link_saved_target(source, target):
                continue
            if (source.review_state == 'unconfirmed' and
                    not has_specific_identity(source) and not has_correction_link and
                    (target.source != 'automatic' or
                     target.review_state != 'unconfirmed' or
                     target.bounds_direct or target.feedback_id is not None)):
                continue
            correction_link = (source.bounds_adjustment is not None or
                               (target.track_id == source.track_id and
                                target.bounds_adjustment is not None and
                                not target.bounds_direct))
            match_coords = target.coords
            if (source.bounds_adjustment is None and
                    target.track_id == source.track_id and
                    target.bounds_adjustment is not None and
                    not target.bounds_direct):
                # Compare the original detector crop when a previous correction
                # was undone. Its expanded crop can have different appearance.
                match_coords = tuple(target.coords[i] - target.bounds_adjustment[i]
                                     for i in range(4))
            if target.source == 'automatic' and not target.bounds_direct and \
                    target.bounds_adjustment is None:
                if target_index not in fitted_targets:
                    fitted_targets[target_index] = _local_foreground_box(
                        current_frame, target.coords)
                fitted = fitted_targets[target_index]
                if (_strong_tracked_fit(reference_frame, current_frame,
                                        source.coords, fitted) and
                        (abs(fitted[2] - target.coords[2]) > 0.1 * target.coords[2] or
                         abs(fitted[3] - target.coords[3]) > 0.1 * target.coords[3])):
                    match_coords = fitted
            overlap = _intersection_over_union(source.coords, match_coords)
            appearance = (_appearance_similarity(reference_frame, current_frame,
                                                 source.coords, match_coords)
                          if overlap >= 0.20 else 0.0)
            edge_match = (0.0 if appearance >= 0.75 else
                          _edge_expansion_similarity(reference_frame, current_frame,
                                                     source.coords, match_coords))
            threshold = (0.65 if correction_link and (
                (source.review_state == 'unconfirmed' and
                 not has_specific_identity(source)) or
                (source.identity_confirmed and target.identity_confirmed and
                 bool(source.pin_id) and source.pin_id == target.pin_id)
            ) else 0.75)
            if (overlap >= 0.20 and appearance >= threshold) or edge_match >= 0.70:
                matches.append((0.6 * max(overlap, 0.35 if edge_match else 0.0) +
                                0.4 * max(appearance, edge_match),
                                source_index, target_index))
    matches.sort(reverse=True)
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    result: list[tuple[BoxData, BoxData]] = []
    for _, source_index, target_index in matches:
        if source_index not in used_sources and target_index not in used_targets:
            result.append((reference_boxes[source_index], current_boxes[target_index]))
            used_sources.add(source_index)
            used_targets.add(target_index)
    return result


def add_new_detections(existing: list[BoxData], detections: list[Coordinate]) -> list[BoxData]:
    """Preserve manual annotations and add only non-overlapping candidates."""
    return add_unique_boxes(existing, [BoxData(coords, [], 'automatic') for coords in detections])


def filter_new_candidates_by_pin_size(image: np.ndarray, existing: list[BoxData],
                                      proposed: list[BoxData]) -> list[BoxData]:
    """Use directly confirmed, fully visible pins as a size prior.

    New proposals should be within 75% to 133% of the median pin width and
    height. A clipped proposal may be narrow along the clipped axis; saved
    annotations and tracked observations are never filtered by this prior.
    """
    height, width = image.shape[:2]
    anchors = [box.coords for box in (*existing, *proposed)
               if box.review_state == 'confirmed' and not box.is_set and
               box.coords[2] > 0 and box.coords[3] > 0 and
               box.coords[0] > 1 and box.coords[1] > 1 and
               box.coords[0] + box.coords[2] < width - 1 and
               box.coords[1] + box.coords[3] < height - 1]
    if not anchors:
        return proposed
    typical_width = float(np.median([box[2] for box in anchors]))
    typical_height = float(np.median([box[3] for box in anchors]))
    typical_area = float(np.median([box[2] * box[3] for box in anchors]))
    minimum_width, maximum_width = 0.75 * typical_width, 4 / 3 * typical_width
    minimum_height, maximum_height = 0.75 * typical_height, 4 / 3 * typical_height
    kept: list[BoxData] = []
    for box in proposed:
        if (box.source != 'automatic' or box.review_state != 'unconfirmed' or
                box.prev_box_id is not None):
            kept.append(box)
            continue
        x, y, w, h = box.coords
        clipped_x = x <= 1 or x + w >= width - 1
        clipped_y = y <= 1 or y + h >= height - 1
        if w <= 0 or h <= 0 or w > maximum_width or h > maximum_height:
            continue
        if ((w < minimum_width and not clipped_x) or
                (h < minimum_height and not clipped_y)):
            continue
        if clipped_x and clipped_y and w * h < 0.10 * typical_area:
            continue
        kept.append(box)
    return kept


def add_unique_boxes(existing: list[BoxData], proposed: list[BoxData]) -> list[BoxData]:
    """Merge detector or tracker output without replacing reviewed boxes."""
    result = list(existing)
    for proposed_box in proposed:
        if all(not _strongly_overlaps(proposed_box.coords, box.coords) for box in result):
            result.append(proposed_box)
    return result


def _same_identified_pin(first: BoxData, second: BoxData) -> bool:
    if first.review_state == 'rejected' or second.review_state == 'rejected':
        return False
    if first.pin_id and second.pin_id:
        return first.pin_id == second.pin_id
    if first.identity_confirmed and second.identity_confirmed and \
            same_specific_name(first, second):
        return True
    return (first.track_id == second.track_id and
            not (has_specific_identity(first) and has_specific_identity(second)
                 and not same_specific_name(first, second)))


def _unreviewed_automatic(box: BoxData) -> bool:
    return (box.source == 'automatic' and box.review_state == 'unconfirmed'
            and box.feedback_id is None and not box.identity_confirmed
            and not box.identity_rejections
            and not (has_specific_identity(box) and box.pin_id is None))


def _box_priority(box: BoxData) -> tuple:
    area = box.coords[2] * box.coords[3]
    return (bool(box.feedback_id), box.identity_confirmed,
            bool(box.identity_rejections),
            box.review_state == 'confirmed', box.source == 'user',
            box.bounds_direct, box.bounds_adjustment is not None,
            has_specific_identity(box) and box.pin_id is None,
            box.review_state == 'inherited', box.review_state == 'rejected',
            has_specific_identity(box), box.match_confidence or 0.0,
            -area if _unreviewed_automatic(box) else 0)


def consolidate_overlapping_boxes(
        boxes: list[BoxData]) -> tuple[list[BoxData], list[BoxData]]:
    """Keep one observation per mostly overlapping candidate or same pin.

    Reviewed boxes keep their object and rectangle. Conflicting reviewed pins
    remain separate so an automatic pass cannot erase a direct decision.
    """
    result = list(boxes)
    removed: list[BoxData] = []
    while True:
        pair = None
        for first_index, first in enumerate(result):
            for second_index in range(first_index + 1, len(result)):
                second = result[second_index]
                if not _strongly_overlaps(first.coords, second.coords):
                    continue
                if (rejects_identity(first, second.pin_id, second.tags) or
                        rejects_identity(second, first.pin_id, first.tags)):
                    continue
                same_pin = _same_identified_pin(first, second)
                if not same_pin and not (_unreviewed_automatic(first) or
                                         _unreviewed_automatic(second)):
                    continue
                if (first.feedback_label and second.feedback_label and
                        first.feedback_label != second.feedback_label):
                    continue
                if (first.identity_confirmed and second.identity_confirmed and
                        first.pin_id and second.pin_id and
                        first.pin_id != second.pin_id):
                    continue
                pair = first_index, second_index, same_pin
                break
            if pair is not None:
                break
        if pair is None:
            return result, removed
        first_index, second_index, same_pin = pair
        first, second = result[first_index], result[second_index]
        winner, loser = (first, second) if _box_priority(first) >= _box_priority(second) \
            else (second, first)
        if same_pin:
            if not has_specific_identity(winner) and has_specific_identity(loser):
                winner.pin_id = loser.pin_id
                winner.tags = list(loser.tags or [])
                winner.match_confidence = loser.match_confidence
                winner.identity_confirmed = loser.identity_confirmed
                winner.identity_inferred = loser.identity_inferred
            winner.identity_rejections = tuple({
                review.example_id: review for review in
                (*winner.identity_rejections, *loser.identity_rejections)
            }.values())
        result[first_index] = winner
        result.pop(second_index)
        removed.append(loser)


def add_adjacent_tracked_boxes(previous_frame: np.ndarray, current_frame: np.ndarray,
                               sources: list[BoxData], existing: list[BoxData],
                               tracked: list[BoxData]) -> list[BoxData]:
    """Retain a tracked review when an unrelated saved proposal overlaps it."""
    matched_tracks = {source.track_id for source, _target in match_existing_candidates(
        previous_frame, current_frame, sources, existing)}
    result = list(existing)
    for box in tracked:
        if box.track_id in matched_tracks or any(
                saved.track_id == box.track_id for saved in result):
            continue
        if any((saved.source == 'user' or
                saved.review_state != 'unconfirmed') and
               _strongly_overlaps(box.coords, saved.coords) for saved in existing):
            continue
        result.append(box)
    return result


def needs_catalog_match(box: BoxData, best_effort: bool = False) -> bool:
    """Allow catalog guesses to improve until a user or tracked identity settles them."""
    if ((box.source not in ('automatic', 'user') and
         not (best_effort and box.source == 'processed')) or
            box.review_state == 'rejected' or box.identity_confirmed):
        return False
    # A typed free-form name is deliberate; do not replace it with a guess.
    if box.source == 'user' and has_specific_identity(box) and box.pin_id is None:
        return False
    if box.review_state in ('confirmed', 'inherited'):
        return True
    return box.source in ('automatic', 'processed') and (
        best_effort or box.pin_id is not None or not has_specific_identity(box))


def identify_candidates(image: np.ndarray, boxes: list[BoxData], matcher: PinMatcher,
                        best_effort: bool = False,
                        feedback_model: FeedbackModel | None = None,
                        prefer_reviewed_identity: bool = False) -> list[BoxData]:
    """Find a catalog suggestion for unverified pin boxes."""
    _validate_image(image)
    height, width = image.shape[:2]
    priority = {'confirmed': 0, 'inherited': 1, 'unconfirmed': 2}
    for box in sorted(boxes, key=lambda item: (priority.get(item.review_state, 3),
                                               item.source != 'user')):
        if not needs_catalog_match(box, best_effort=best_effort):
            continue
        x, y, w, h = box.coords
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            continue
        matcher_method = (getattr(matcher, 'suggest', matcher.match)
                          if best_effort or box.review_state in ('confirmed', 'inherited')
                          else matcher.match)
        exclusions = box.identity_rejections
        excluded_ids = frozenset(item.pin_id for item in exclusions if item.pin_id)
        excluded_names = frozenset(
            tag.casefold() for item in exclusions if item.pin_id is None
            for tag in item.tags)
        if exclusions:
            match = matcher_method(
                image[y:y + h, x:x + w],
                excluded_pin_ids=excluded_ids, excluded_names=excluded_names)
        else:
            match = matcher_method(image[y:y + h, x:x + w])
        if (feedback_model is not None and
                (box.review_state in ('confirmed', 'inherited') or
                 (best_effort and prefer_reviewed_identity))):
            learned = feedback_model.suggest_identity(
                image[y:y + h, x:x + w], excluded_ids, excluded_names)
            if learned is not None and (match is None or
                    (getattr(match, 'inliers', 0) == 0 and
                     (prefer_reviewed_identity or
                      (match.confidence or 0.0) < 0.60))):
                match = learned
        if match is not None:
            box.pin_id = match.pin_id
            box.tags = [match.name]
            box.match_confidence = match.confidence if match.confidence is not None else 0.0
    return boxes


def inherit_confirmed_catalog_matches(previous_boxes: list[BoxData],
                                      current_boxes: list[BoxData]) -> int:
    """Link nearby catalog matches when appearance tracking missed a verified pin.

    An independent geometric catalog match and substantial overlap must agree.
    This does not turn a suggestion into a direct user confirmation.
    """
    choices: list[tuple[float, int, int]] = []
    for source_index, source in enumerate(previous_boxes):
        if (source.review_state not in ('confirmed', 'inherited') or
                not source.identity_confirmed or source.identity_inferred or
                not has_specific_identity(source)):
            continue
        for target_index, target in enumerate(current_boxes):
            same_identity = (target.pin_id == source.pin_id if source.pin_id else
                             same_specific_name(source, target))
            if (target.source != 'automatic' or target.review_state != 'unconfirmed' or
                    target.identity_confirmed or not target.pin_id or not same_identity or
                    target.match_confidence is None or target.match_confidence < 0.60 or
                    rejects_identity(target, source.pin_id, source.tags)):
                continue
            overlap = _intersection_over_union(source.coords, target.coords)
            if overlap >= 0.55:
                choices.append((overlap, source_index, target_index))
    choices.sort(reverse=True)
    used_sources: set[int] = set()
    used_targets: set[int] = set()
    for _overlap, source_index, target_index in choices:
        if source_index in used_sources or target_index in used_targets:
            continue
        source, target = previous_boxes[source_index], current_boxes[target_index]
        target.track_id = source.track_id
        target.prev_box_id = source.id
        target.review_state = 'inherited'
        target.identity_confirmed = True
        target.identity_inferred = False
        target.pin_id = source.pin_id
        target.tags = list(source.tags or [])
        target.identity_rejections = tuple({item.example_id: item for item in (
            *target.identity_rejections, *source.identity_rejections)}.values())
        used_sources.add(source_index)
        used_targets.add(target_index)
    return len(used_targets)
