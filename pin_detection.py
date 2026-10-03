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


def _strongly_overlaps(a: Coordinate, b: Coordinate) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return _intersection_over_union(a, b) >= 0.35 or intersection / min(aw * ah, bw * bh) >= 0.65


def _has_local_contrast(image: np.ndarray, region: Coordinate) -> bool:
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
    return difference >= 15.0


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
        if max(w, h) / min(w, h) > 3.0 or cv2.contourArea(contour) < min_side * min_side * 0.3:
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
        if min(w, h) < min_side or max(w, h) > max_side or max(w, h) / min(w, h) > 3:
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
        if (min(w, h) < min_side or max(w, h) > max_side or
                max(w, h) / min(w, h) > 2.5 or
                cv2.contourArea(contour) < 0.25 * min_side * min_side):
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
    regions = regions[:200]
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


def _search_for_previous_box(previous: np.ndarray, current: np.ndarray, coords: Coordinate) -> Coordinate | None:
    """Search a small neighborhood for an unchanged pin when proposals miss it."""
    x, y, w, h = coords
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > previous.shape[1] or y + h > previous.shape[0]:
        return None
    template = previous[y:y + h, x:x + w]
    if float(np.std(template.astype(np.float32), axis=(0, 1)).mean()) < 8:
        return None
    padding = int(0.75 * max(w, h))
    left, top = max(0, x - padding), max(0, y - padding)
    right = min(current.shape[1], x + w + padding)
    bottom = min(current.shape[0], y + h + padding)
    search = current[top:bottom, left:right]
    if search.shape[0] < h or search.shape[1] < w:
        return None
    result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    candidate = (left + location[0], top + location[1], w, h)
    if score < 0.82 or _appearance_similarity(previous, current, coords, candidate) < 0.65:
        return None
    return candidate


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
    for old_index, old_box in enumerate(previous_boxes):
        for new_index, detection in enumerate(detections):
            overlap = _intersection_over_union(old_box.coords, detection)
            if overlap < 0.20:
                continue
            appearance = _appearance_similarity(previous_frame, current_frame, old_box.coords, detection)
            if appearance >= 0.65:
                matches.append((old_box.review_state != 'rejected',
                                0.6 * overlap + 0.4 * appearance, old_index, new_index))
    matches.sort(reverse=True)
    assigned_old: set[int] = set()
    assigned_new: dict[int, BoxData] = {}
    for _, _, old_index, new_index in matches:
        if old_index in assigned_old or new_index in assigned_new:
            continue
        old_box = previous_boxes[old_index]
        state = ('rejected' if old_box.review_state == 'rejected' else
                 'inherited' if old_box.review_state != 'unconfirmed' else 'unconfirmed')
        tracked = BoxData(detections[new_index],
                          (list(old_box.tags or []) if state != 'rejected' and
                           has_specific_identity(old_box) else []),
                          'automatic', old_box.track_id,
                          None if state == 'rejected' else old_box.pin_id, state,
                          match_confidence=(None if state == 'rejected' else
                                            old_box.match_confidence),
                          identity_confirmed=(False if state == 'rejected' else
                                              old_box.identity_confirmed),
                          identity_rejections=old_box.identity_rejections)
        tracked.prev_box_id = old_box.id
        assigned_old.add(old_index)
        assigned_new[new_index] = tracked
    result = [assigned_new.get(index) or BoxData(coords, [], 'automatic')
              for index, coords in enumerate(detections)]
    for old_index, old_box in enumerate(previous_boxes):
        if old_index in assigned_old or (not has_specific_identity(old_box)
                                         and old_box.review_state == 'unconfirmed'):
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
                          identity_rejections=old_box.identity_rejections)
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
        return False
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
    for source_index, source in enumerate(reference_boxes):
        if source.review_state == 'rejected' or not (
                source.review_state in ('confirmed', 'inherited') or
                has_specific_identity(source)):
            continue
        for target_index, target in enumerate(current_boxes):
            if not can_link_saved_target(source, target):
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
            result.append((reference_boxes[source_index], current_boxes[target_index]))
            used_sources.add(source_index)
            used_targets.add(target_index)
    return result


def add_new_detections(existing: list[BoxData], detections: list[Coordinate]) -> list[BoxData]:
    """Preserve manual annotations and add only non-overlapping candidates."""
    return add_unique_boxes(existing, [BoxData(coords, [], 'automatic') for coords in detections])


def add_unique_boxes(existing: list[BoxData], proposed: list[BoxData]) -> list[BoxData]:
    """Merge detector or tracker output without replacing reviewed boxes."""
    result = list(existing)
    for proposed_box in proposed:
        if all(not _strongly_overlaps(proposed_box.coords, box.coords) for box in result):
            result.append(proposed_box)
    return result


def needs_catalog_match(box: BoxData, best_effort: bool = False) -> bool:
    """Allow catalog guesses to improve until a user or tracked identity settles them."""
    if best_effort:
        return (box.source in ('automatic', 'user') and
                box.review_state != 'rejected' and not box.identity_confirmed)
    eligible_source = (box.source == 'automatic' or
                       (box.source == 'user' and box.review_state == 'confirmed' and
                        not has_specific_identity(box)))
    return (eligible_source and box.review_state not in ('inherited', 'rejected')
            and not box.identity_confirmed
            and (box.pin_id is not None or not has_specific_identity(box)))


def identify_candidates(image: np.ndarray, boxes: list[BoxData], matcher: PinMatcher,
                        best_effort: bool = False) -> list[BoxData]:
    """Find a catalog suggestion for unverified pin boxes."""
    _validate_image(image)
    height, width = image.shape[:2]
    for box in boxes:
        if not needs_catalog_match(box, best_effort=best_effort):
            continue
        x, y, w, h = box.coords
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
            continue
        matcher_method = (getattr(matcher, 'suggest', matcher.match)
                          if best_effort or (box.source == 'user' and
                                             box.review_state == 'confirmed')
                          else matcher.match)
        exclusions = box.identity_rejections
        if exclusions:
            match = matcher_method(
                image[y:y + h, x:x + w],
                excluded_pin_ids=frozenset(item.pin_id for item in exclusions if item.pin_id),
                excluded_names=frozenset(
                    tag.casefold() for item in exclusions if item.pin_id is None
                    for tag in item.tags))
        else:
            match = matcher_method(image[y:y + h, x:x + w])
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
