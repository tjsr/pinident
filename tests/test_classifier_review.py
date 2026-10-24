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


def test_saved_review_areas_start_at_current_frame_and_defer_confirmed() -> None:
    before = BoxData((0, 0, 10, 10), [], 'automatic')
    current_confirmed = BoxData((0, 0, 10, 10), ['Known'], 'automatic',
                                review_state='confirmed')
    current_candidate = BoxData((12, 0, 10, 10), [], 'automatic')
    later_linked = BoxData((0, 0, 10, 10), ['Known'], 'automatic',
                           review_state='inherited')
    later_confirmed = BoxData((12, 0, 10, 10), [], 'automatic',
                              review_state='confirmed')
    current_rejected = BoxData((24, 0, 10, 10), [], 'automatic',
                               review_state='rejected')

    areas = saved_review_areas({
        1: [before],
        5: [current_confirmed, current_candidate, current_rejected],
        9: [later_linked, later_confirmed],
    }, start_frame=5)

    assert [(area.frame_index, area.box) for area in areas] == [
        (5, current_candidate), (9, later_linked), (1, before),
        (5, current_confirmed), (9, later_confirmed), (5, current_rejected)]


def test_saved_review_areas_rank_strength_before_confirmed_pin_count() -> None:
    common = BoxData((0, 0, 10, 10), ['Common'], 'automatic',
                     pin_id='pin-common', match_confidence=0.90)
    rare = BoxData((12, 0, 10, 10), ['Rare'], 'automatic',
                   pin_id='pin-rare', match_confidence=0.90)
    strongest = BoxData((24, 0, 10, 10), ['Common'], 'automatic',
                        pin_id='pin-common', match_confidence=0.96)
    unscored = BoxData((36, 0, 10, 10), [], 'automatic')
    direct_common = BoxData((0, 0, 10, 10), ['Common'], 'automatic',
                            pin_id='pin-common', review_state='confirmed',
                            identity_confirmed=True)
    inferred_rare = BoxData((12, 0, 10, 10), ['Rare'], 'automatic',
                            pin_id='pin-rare', review_state='confirmed',
                            identity_confirmed=True, identity_inferred=True)
    confirmed_common = BoxData((24, 0, 10, 10), ['Common'], 'automatic',
                               pin_id='pin-common', review_state='confirmed',
                               identity_confirmed=True)

    areas = saved_review_areas({
        1: [direct_common, inferred_rare],
        5: [common, unscored, rare, strongest, confirmed_common],
    }, start_frame=5)

    assert [area.box for area in areas] == [
        strongest, rare, unscored, common,
        confirmed_common, direct_common, inferred_rare]


def test_saved_review_areas_show_each_suggested_pin_before_repeats() -> None:
    first_a = BoxData((0, 0, 10, 10), ['A'], 'automatic',
                      pin_id='pin-a', match_confidence=0.99)
    second_a = BoxData((12, 0, 10, 10), ['A'], 'automatic',
                       pin_id='pin-a', match_confidence=0.98)
    first_b = BoxData((24, 0, 10, 10), ['B'], 'automatic',
                      pin_id='pin-b', match_confidence=0.97)
    third_a = BoxData((36, 0, 10, 10), ['A'], 'automatic',
                      pin_id='pin-a', match_confidence=0.96)
    second_b = BoxData((48, 0, 10, 10), ['B'], 'automatic',
                       pin_id='pin-b', match_confidence=0.95)
    unidentified = BoxData((60, 0, 10, 10), [], 'automatic')
    confirmed = BoxData((0, 12, 10, 10), ['A'], 'automatic',
                        pin_id='pin-a', review_state='confirmed')
    rejected = BoxData((12, 12, 10, 10), ['A'], 'automatic',
                       pin_id='pin-a', review_state='rejected')

    areas = saved_review_areas({0: [
        first_a, second_a, first_b, third_a, second_b, unidentified,
        confirmed, rejected]})

    assert [area.box for area in areas] == [
        first_a, first_b, unidentified, second_a, second_b, third_a,
        confirmed, rejected]


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
