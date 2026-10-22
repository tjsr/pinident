from boxdata import BoxData
from classifier_review import review_actions, review_kind, saved_review_areas


def test_saved_review_areas_defer_rejected_boxes_until_active_areas_are_done() -> None:
    first = BoxData((1, 2, 10, 10), [], 'automatic')
    second = BoxData((3, 4, 10, 10), ['Pin'], 'automatic',
                     review_state='rejected')
    later = BoxData((5, 6, 10, 10), [], 'user', review_state='confirmed')

    areas = saved_review_areas({7: [later], 2: [first, second]})

    assert [(area.frame_index, area.box) for area in areas] == [
        (2, first), (7, later), (2, second)]


def test_review_states_and_actions_preserve_presence_identity_distinction() -> None:
    candidate = BoxData((0, 0, 20, 20), [], 'automatic')
    suggested = BoxData((0, 0, 20, 20), ['Guess'], 'automatic', pin_id='pin-1')
    present = BoxData((0, 0, 20, 20), [], 'user', review_state='confirmed')
    linked = BoxData((0, 0, 20, 20), ['Known'], 'automatic', pin_id='pin-2',
                     review_state='inherited', identity_confirmed=True)
    verified = BoxData((0, 0, 20, 20), ['Known'], 'automatic', pin_id='pin-2',
                       review_state='confirmed', identity_confirmed=True)
    inferred = BoxData((0, 0, 20, 20), ['Known'], 'automatic', pin_id='pin-2',
                       review_state='confirmed', identity_confirmed=True,
                       identity_inferred=True)
    rejected = BoxData((0, 0, 20, 20), ['Wrong'], 'automatic',
                       review_state='rejected')

    assert [review_kind(box) for box in (
        candidate, suggested, present, linked, verified, rejected)] == [
        'candidate', 'suggested_candidate', 'confirmed_pin', 'linked_pin',
        'confirmed_identity', 'rejected']
    assert review_actions(candidate) == frozenset({
        'confirm_pin', 'choose_identity', 'not_pin'})
    assert 'reject_identity' in review_actions(suggested)
    assert 'not_pin' not in review_actions(present)
    assert 'confirm_identity' in review_actions(linked)
    assert 'confirm_pin' not in review_actions(present)
    assert review_kind(inferred) == 'linked_pin'
    assert 'reject_identity' in review_actions(verified)
    assert review_actions(rejected) == frozenset({'restore_pin'})
