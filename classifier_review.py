"""Saved-rectangle ordering and review states for the desktop review browser."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from boxdata import BoxData, has_specific_identity
from false_positive_log import collect_false_positives, false_positive_counts


ReviewKind = Literal[
    'candidate', 'suggested_candidate', 'confirmed_pin',
    'linked_pin', 'confirmed_identity', 'rejected',
]


@dataclass(frozen=True)
class ReviewArea:
    frame_index: int
    box: BoxData


def review_priority(box: BoxData) -> int:
    """Leave direct confirmations and rejected areas until other reviews finish."""
    return 2 if box.review_state == 'rejected' else (
        1 if box.review_state == 'confirmed' else 0)


def confirmed_identity_counts(areas: Iterable[ReviewArea]) -> dict[str, int]:
    """Count distinct directly verified observations of each catalog pin."""
    counts: dict[str, int] = {}
    for area in areas:
        box = area.box
        if (box.review_state == 'confirmed' and box.identity_confirmed and
                not box.identity_inferred and box.pin_id):
            counts[box.pin_id] = counts.get(box.pin_id, 0) + 1
    return counts


def review_sort_key(area: ReviewArea,
                    confirmed_counts: dict[str, int],
                    incorrect_counts: dict[str, int]) -> tuple[int, int, float, int]:
    """Prioritize IDs with more direct errors, then strength and sparse reviews."""
    box = area.box
    tier = review_priority(box)
    if tier:
        return tier, 0, 0.0, 0
    strength = box.match_confidence
    score = (-float(strength) if isinstance(strength, (int, float))
             and not isinstance(strength, bool) and isfinite(strength)
             and 0 <= strength <= 1 else 1.0)
    return (tier, -incorrect_counts.get(box.pin_id, 0) if box.pin_id else 0,
            score,
            confirmed_counts.get(box.pin_id, 0) if box.pin_id else 0)


def incorrect_identity_counts(areas: Iterable[ReviewArea]) -> dict[str, int]:
    grouped: dict[int, list[BoxData]] = {}
    for area in areas:
        grouped.setdefault(area.frame_index, []).append(area.box)
    return false_positive_counts(collect_false_positives(grouped))


def order_review_areas(areas: Iterable[ReviewArea],
                       confirmed_counts: dict[str, int],
                       shown_pin_ids: Iterable[str] = (),
                       incorrect_counts: dict[str, int] | None = None) -> list[ReviewArea]:
    """Rank areas, then show each suggested ID once before its next round."""
    remaining = list(areas)
    errors = (incorrect_identity_counts(remaining) if incorrect_counts is None
              else incorrect_counts)
    ranked = sorted(remaining, key=lambda area: review_sort_key(
        area, confirmed_counts, errors))
    occurrences: dict[str, int] = {}
    for pin_id in shown_pin_ids:
        occurrences[pin_id] = occurrences.get(pin_id, 0) + 1
    rounds: list[list[ReviewArea]] = []
    deferred: list[ReviewArea] = []
    for area in ranked:
        if review_priority(area.box):
            deferred.append(area)
            continue
        pin_id = area.box.pin_id
        round_index = occurrences.get(pin_id, 0) if pin_id else 0
        if pin_id:
            occurrences[pin_id] = round_index + 1
        while len(rounds) <= round_index:
            rounds.append([])
        rounds[round_index].append(area)
    return [area for group in rounds for area in group] + deferred


def saved_review_areas(frame_boxes: dict[int, list[BoxData]],
                       start_frame: int = 0) -> list[ReviewArea]:
    """Rank review work, using current-frame order to resolve equal priorities."""
    frames = sorted(frame_boxes, key=lambda index: (index < start_frame, index))
    ordered = [ReviewArea(index, box)
               for index in frames for box in frame_boxes[index]]
    counts = confirmed_identity_counts(ordered)
    return order_review_areas(ordered, counts)


def review_kind(box: BoxData) -> ReviewKind:
    if box.review_state == 'rejected':
        return 'rejected'
    if box.review_state == 'inherited':
        return 'linked_pin'
    if box.review_state == 'confirmed':
        if box.identity_inferred:
            return 'linked_pin'
        return 'confirmed_identity' if box.identity_confirmed else 'confirmed_pin'
    return 'suggested_candidate' if has_specific_identity(box) else 'candidate'


def review_actions(box: BoxData) -> frozenset[str]:
    """Return review decisions supported by the existing annotation contract."""
    if box.review_state == 'rejected':
        return frozenset({'restore_pin'}) if box.source == 'automatic' else frozenset()
    actions = {'choose_identity'}
    if box.review_state in ('unconfirmed', 'inherited'):
        actions.add('confirm_pin')
    if has_specific_identity(box):
        actions.update(('confirm_identity', 'reject_identity'))
    if box.source == 'automatic':
        actions.add('not_pin')
    return frozenset(actions)
