from pathlib import Path

from boxdata import BoxData, IdentityRejection
from classifier_review import saved_review_areas
from false_positive_log import (collect_false_positives, false_positive_counts,
                                false_positive_path, load_false_positives)
from scrubberframe import save_boxes_to_file


def test_false_positive_log_deduplicates_linked_reviews_and_updates_on_save(
        tmp_path: Path) -> None:
    filename = str(tmp_path / 'video.json')
    review = IdentityRejection('wrong-1', 'pin-a', ('A',), (1, 2, 10, 10), 0)
    first = BoxData((1, 2, 10, 10), [], 'automatic',
                    identity_rejections=(review,))
    linked = BoxData((2, 2, 10, 10), [], 'automatic',
                     identity_rejections=(review,))
    rejected = BoxData((20, 2, 10, 10), ['A'], 'automatic',
                       pin_id='pin-a', review_state='rejected',
                       feedback_id='negative-1', feedback_label='negative')
    boxes = {0: [first, rejected], 1: [linked]}

    save_boxes_to_file(filename, boxes)

    records = load_false_positives(filename)
    assert records is not None
    assert false_positive_path(filename).is_file()
    assert [(item.review_id, item.kind, item.frame_index, item.pin_id)
            for item in records] == [
                ('wrong-1', 'wrong_identity', 0, 'pin-a'),
                ('negative-1', 'not_pin', 0, 'pin-a')]
    assert false_positive_counts(records) == {'pin-a': 2}

    first.identity_rejections = ()
    linked.identity_rejections = ()
    rejected.review_state = 'confirmed'
    rejected.feedback_label = 'positive'
    save_boxes_to_file(filename, boxes)
    assert load_false_positives(filename) == ()


def test_false_positive_frequency_prioritizes_review_candidates() -> None:
    frequent = BoxData((0, 0, 10, 10), ['A'], 'automatic',
                       pin_id='pin-a', match_confidence=0.6)
    strong = BoxData((12, 0, 10, 10), ['B'], 'automatic',
                     pin_id='pin-b', match_confidence=0.99)
    repeat = BoxData((24, 0, 10, 10), ['A'], 'automatic',
                     pin_id='pin-a', match_confidence=0.5)
    previous = BoxData((36, 0, 10, 10), [], 'automatic',
                       identity_rejections=(
                           IdentityRejection('wrong-1', 'pin-a', ('A',),
                                             (36, 0, 10, 10), 0),
                           IdentityRejection('wrong-2', 'pin-a', ('A',),
                                             (36, 0, 10, 10), 0)))
    assert false_positive_counts(collect_false_positives({0: [previous]})) == {
        'pin-a': 2}

    areas = saved_review_areas({0: [strong, frequent, repeat, previous]})
    assert [area.box for area in areas] == [frequent, strong, previous, repeat]
