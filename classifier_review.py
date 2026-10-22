"""Saved-rectangle ordering and review states for the desktop review browser."""

from dataclasses import dataclass
from typing import Literal

from boxdata import BoxData, has_specific_identity


ReviewKind = Literal[
    'candidate', 'suggested_candidate', 'confirmed_pin',
    'linked_pin', 'confirmed_identity', 'rejected',
]


@dataclass(frozen=True)
class ReviewArea:
    frame_index: int
    box: BoxData


def saved_review_areas(frame_boxes: dict[int, list[BoxData]]) -> list[ReviewArea]:
    """Review active areas first, then previously rejected areas."""
    ordered = [ReviewArea(index, box)
               for index in sorted(frame_boxes)
               for box in frame_boxes[index]]
    return ([area for area in ordered if area.box.review_state != 'rejected'] +
            [area for area in ordered if area.box.review_state == 'rejected'])


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
