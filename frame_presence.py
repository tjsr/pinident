"""Per-frame user assertions and derived pin presence."""

from dataclasses import dataclass

from boxdata import BoxData, has_specific_identity


@dataclass(frozen=True)
class FramePresence:
    pin_asserted: bool = False
    set_asserted: bool = False

    @property
    def minimum_pins(self) -> int:
        return 2 if self.set_asserted else 1 if self.pin_asserted else 0


def marked_pin_count(boxes: list[BoxData]) -> int:
    return len({box.track_id for box in boxes if box.review_state != 'rejected' and (
        box.review_state in ('confirmed', 'inherited') or has_specific_identity(box))})


def checkbox_values(presence: FramePresence, boxes: list[BoxData]) -> tuple[bool, bool]:
    count = marked_pin_count(boxes)
    contains_set = presence.set_asserted or count >= 2
    return presence.pin_asserted or contains_set or count >= 1, contains_set
