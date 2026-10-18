import cv2
import json
import numpy as np
import pytest
import wx
from io import StringIO
from time import monotonic, sleep
from unittest.mock import patch

from boxdata import BoxData, IdentityRejection
from detection_feedback import FeedbackModel, FeedbackStore
from frame_processing import FrameJob, FrameResult, process_frame
from pin_detection import (add_new_detections, add_unique_boxes, associate_frame_boxes,
                           consolidate_overlapping_boxes,
                           detect_pin_boxes, detect_pin_boxes_relaxed,
                           filter_new_candidates_by_pin_size,
                           inherit_corrected_bounds, match_existing_candidates,
                           refine_region)
from scrubberframe import (load_boxes_from_stream, reconcile_saved_frames,
                           save_boxes_to_stream)
from videoscrubber import VideoScrubber


BACKGROUND = (145, 165, 185)


def make_frame(first_x=45, include_second=True):
    frame = np.full((240, 360, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(frame, (first_x, 65), (first_x + 55, 125), (25, 35, 190), -1)
    cv2.rectangle(frame, (first_x, 65), (first_x + 55, 125), (10, 10, 10), 3)
    if include_second:
        cv2.circle(frame, (245, 115), 32, (210, 55, 30), -1)
        cv2.circle(frame, (245, 115), 32, (5, 5, 5), 3)
    return frame


def center_inside(box, point):
    x, y, w, h = box
    return x <= point[0] < x + w and y <= point[1] < y + h


def test_coarse_region_is_refined_to_the_object():
    frame = make_frame(include_second=False)
    assert refine_region(frame, (28, 49, 90, 93)) == (43, 63, 60, 65)


def test_detects_two_separate_candidates():
    boxes = detect_pin_boxes(make_frame())
    assert len(boxes) == 2
    assert any(center_inside(box, (72, 95)) for box in boxes)
    assert any(center_inside(box, (245, 115)) for box in boxes)


def test_relaxed_search_finds_low_contrast_object_missed_by_primary():
    frame = np.full((240, 360, 3), 125, dtype=np.uint8)
    cv2.circle(frame, (130, 120), 34, (137, 137, 137), -1)
    cv2.circle(frame, (130, 120), 34, (113, 113, 113), 2)
    assert detect_pin_boxes(frame) == []
    assert any(center_inside(box, (130, 120))
               for box in detect_pin_boxes_relaxed(frame))


def test_relaxed_search_rejects_repeating_low_contrast_texture():
    frame = np.full((240, 360, 3), 125, dtype=np.uint8)
    for y in range(20, 220, 40):
        for x in range(20, 340, 40):
            cv2.circle(frame, (x, y), 11, (133, 133, 133), -1)
            cv2.circle(frame, (x, y), 11, (117, 117, 117), 1)

    assert detect_pin_boxes_relaxed(frame) == []


def test_binary_classifier_cannot_erase_plausible_pin_proposals():
    class AlwaysNegative:
        def feature(self, _crop):
            return np.ones(1, dtype=np.float32)

        def classify_feature(self, _feature):
            return 'negative'

    store = FeedbackStore()
    try:
        feedback = FeedbackModel(store)
        feedback.trained_model = AlwaysNegative()
        frame = make_frame()
        assert len(detect_pin_boxes(frame)) == 2
        assert len(detect_pin_boxes(frame, feedback_model=feedback)) == 2
    finally:
        store.close()


def test_detection_does_not_replace_a_manual_box():
    manual = BoxData((43, 63, 60, 65), ['Mario'], 'user')
    boxes = add_new_detections([manual], detect_pin_boxes(make_frame()))

    assert len(boxes) == 2
    assert boxes[0] is manual
    assert boxes[1].tags == []


def test_new_candidate_sizes_follow_confirmed_pins_but_keep_edge_fragments():
    frame = np.full((240, 360, 3), BACKGROUND, dtype=np.uint8)
    confirmed = BoxData((45, 50, 60, 60), ['Mario'], 'user',
                        review_state='confirmed')
    proposals = [BoxData(coords, [], 'automatic') for coords in (
        (150, 50, 62, 58), (255, 50, 9, 10), (130, 120, 170, 105),
        (0, 155, 15, 58), (0, 205, 9, 10))]

    kept = filter_new_candidates_by_pin_size(frame, [confirmed], proposals)

    assert [box.coords for box in kept] == [(150, 50, 62, 58), (0, 155, 15, 58)]


def test_size_prior_uses_75_to_133_percent_on_each_full_dimension():
    frame = np.full((240, 500, 3), BACKGROUND, dtype=np.uint8)
    confirmed = BoxData((20, 20, 60, 60), [], 'user', review_state='confirmed')
    proposed = [BoxData(coords, [], 'automatic') for coords in (
        (110, 20, 45, 45), (170, 20, 80, 80),
        (270, 20, 44, 60), (330, 20, 81, 60),
        (110, 125, 60, 44), (190, 125, 60, 81),
        (0, 125, 20, 60), (270, 0, 60, 20),
        (0, 0, 10, 10), (0, 125, 20, 44))]

    kept = filter_new_candidates_by_pin_size(frame, [confirmed], proposed)

    assert [box.coords for box in kept] == [
        (110, 20, 45, 45), (170, 20, 80, 80),
        (0, 125, 20, 60), (270, 0, 60, 20)]


def test_size_filter_preserves_saved_and_tracked_boxes_and_needs_a_pin_reference():
    frame = np.full((240, 360, 3), BACKGROUND, dtype=np.uint8)
    saved = BoxData((130, 120, 170, 105), [], 'automatic')
    candidate = BoxData((255, 50, 9, 10), [], 'automatic')
    assert filter_new_candidates_by_pin_size(frame, [saved], [candidate]) == [candidate]

    set_box = BoxData((30, 30, 220, 160), ['Pin set'], 'user',
                      review_state='confirmed')
    set_box.is_set = True
    assert filter_new_candidates_by_pin_size(frame, [set_box], [candidate]) == [candidate]

    inherited = BoxData((45, 50, 60, 60), [], 'automatic',
                        review_state='inherited', identity_confirmed=True)
    assert filter_new_candidates_by_pin_size(frame, [inherited], [candidate]) == [candidate]

    confirmed = BoxData((45, 50, 60, 60), [], 'user',
                        review_state='confirmed')
    tracked = BoxData((130, 120, 170, 105), ['Mario'], 'automatic',
                      review_state='inherited')
    assert filter_new_candidates_by_pin_size(frame, [confirmed, saved],
                                             [tracked, candidate]) == [tracked]


def test_worker_filters_primary_and_fallback_candidates_using_confirmed_pin_size():
    frame = np.full((240, 360, 3), BACKGROUND, dtype=np.uint8)
    confirmed = BoxData((45, 50, 60, 60), [], 'user', review_state='confirmed')

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(0, None, [], [confirmed], [], frame, None, None, NoMatch(),
                   expected_pins=2)
    with (patch('frame_processing.detect_pin_boxes', return_value=[
            (150, 50, 62, 58), (255, 50, 9, 10)]),
          patch('frame_processing.detect_pin_boxes_relaxed', return_value=[
              (130, 120, 170, 105), (0, 155, 15, 58)])):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert [box.coords for box in result.boxes] == [
        confirmed.coords, (150, 50, 62, 58), (0, 155, 15, 58)]


def test_worker_uses_tighter_size_prior_for_primary_and_relaxed_proposals():
    frame = np.full((240, 500, 3), BACKGROUND, dtype=np.uint8)
    confirmed = BoxData((20, 20, 60, 60), [], 'user', review_state='confirmed')

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(0, None, [], [confirmed], [], frame, None, None, NoMatch(),
                   expected_pins=2)
    with (patch('frame_processing.detect_pin_boxes', return_value=[
            (110, 20, 44, 60), (180, 20, 45, 60), (250, 20, 80, 60)]),
          patch('frame_processing.detect_pin_boxes_relaxed', return_value=[
              (110, 125, 60, 44), (200, 125, 60, 45), (290, 125, 60, 81)])):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert [box.coords for box in result.boxes] == [
        confirmed.coords, (180, 20, 45, 60), (250, 20, 80, 60),
        (200, 125, 60, 45)]


def test_fallback_matches_only_new_boxes_once(caplog):
    frame = np.full((240, 360, 3), BACKGROUND, dtype=np.uint8)
    existing = BoxData((40, 40, 40, 40), [], 'automatic')
    job = FrameJob(0, None, [], [existing], [], frame, None, None, None,
                   expected_pins=2)
    matched = []

    def record_matches(_frame, boxes, _matcher, **_kwargs):
        matched.append([box.coords for box in boxes])
        return boxes

    with (patch('frame_processing.detect_pin_boxes', return_value=[(130, 40, 40, 40)]),
          patch('frame_processing.detect_pin_boxes_relaxed',
                return_value=[(220, 40, 40, 40)]),
          patch('frame_processing.identify_candidates', side_effect=record_matches)):
        process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert matched == [[existing.coords, (130, 40, 40, 40)],
                       [(220, 40, 40, 40)]]
    worker_log = next(record.message for record in caplog.records
                      if 'worker decode=' in record.message)
    assert 'fallback_detect=' in worker_log
    assert 'fallback_catalog=' in worker_log


def test_explicit_detection_logs_source_proposals(caplog):
    class NoMatch:
        def match(self, _crop):
            return None

    frame = make_frame(include_second=False)
    proposal = (43, 63, 60, 65)
    job = FrameJob(0, None, [], [], [], frame, None, None, NoMatch(),
                   diagnostic_run_id='diagnose1')
    with patch('frame_processing.detect_pin_boxes', return_value=[proposal]):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    worker_line = next(record.message.split('worker ', 1)[1]
                       for record in caplog.records
                       if 'PINIDENT_DETECTION worker ' in record.message)
    worker = json.loads(worker_line)
    assert worker['run'] == 'diagnose1'
    assert worker['source_size'] == [360, 240]
    assert worker['primary_xywh'] == [list(proposal)]
    assert worker['result_count'] == 1
    assert result.diagnostic_run_id == 'diagnose1'


@pytest.mark.gui
def test_detection_diagnostic_logs_panel_projection_and_suggestion(caplog):
    app = wx.App.Get() or wx.App(False)
    frame = make_frame(include_second=False)
    window = VideoScrubber(None, 'Detection diagnostics', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((43, 63, 60, 65), ['Suggested Mario'], 'automatic',
                      pin_id='pinpanion:7', match_confidence=0.72)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(frame, reconcile=False)
        result = FrameResult(0, frame, [box], 0, 0.0,
                             diagnostic_run_id='diagnose2')
        window._log_detection_display(result)

        display_line = next(record.message.split('display ', 1)[1]
                            for record in caplog.records
                            if 'PINIDENT_DETECTION display ' in record.message)
        box_line = next(record.message.split('box ', 1)[1]
                        for record in caplog.records
                        if 'PINIDENT_DETECTION box ' in record.message)
        display = json.loads(display_line)
        logged_box = json.loads(box_line)
        scale_x = display['bitmap_size'][0] / display['display_size'][0]
        scale_y = display['bitmap_size'][1] / display['display_size'][1]
        assert display['source_size'] == [360, 240]
        assert display['rotation'] == 0
        assert logged_box['source_xywh'] == [43, 63, 60, 65]
        assert logged_box['panel_xywh'][:2] == [
            display['bitmap_offset'][0] + round(43 * scale_x),
            display['bitmap_offset'][1] + round(63 * scale_y)]
        assert logged_box['origin'] == 'new'
        assert logged_box['in_source_bounds']
        assert logged_box['tags'] == ['Suggested Mario']
        assert logged_box['pin_id'] == 'pinpanion:7'
        assert logged_box['match_confidence'] == 0.72
        with patch.object(window, '_submit_frame_job') as submit:
            window.on_detect_async(None)
        assert submit.call_args.kwargs['diagnostic_run_id']
        assert submit.call_args.kwargs['catalog_best_effort']
    finally:
        window.Destroy()


def test_panning_frame_searches_for_new_edge_pin_beside_tracked_pin():
    previous = np.full((240, 360, 3), 125, dtype=np.uint8)
    current = previous.copy()
    for frame, x in ((previous, 95), (current, 65)):
        cv2.rectangle(frame, (x, 65), (x + 55, 125), (25, 35, 190), -1)
        cv2.rectangle(frame, (x, 65), (x + 55, 125), (10, 10, 10), 3)
    cv2.circle(current, (300, 120), 34, (137, 137, 137), -1)
    cv2.circle(current, (300, 120), 34, (113, 113, 113), 2)
    old_box = BoxData(detect_pin_boxes(previous)[0], ['Known'], 'automatic',
                      track_id='known', review_state='confirmed',
                      identity_confirmed=True)

    class Matcher:
        attempts = 0

        def match(self, _crop):
            self.attempts += 1
            return None

    matcher = Matcher()
    job = FrameJob(1, 0, [old_box], [], [], current, previous, None, matcher,
                   allow_motion_fallback=True)
    result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))
    assert result.primary_count == 1
    assert result.relaxed_count > 0
    assert any(center_inside(box.coords, (300, 120)) for box in result.boxes)
    assert matcher.attempts >= 1


def test_static_playback_does_not_run_motion_search():
    frame = make_frame(include_second=False)
    old_box = BoxData(detect_pin_boxes(frame)[0], ['Known'], 'automatic',
                      review_state='confirmed', identity_confirmed=True)

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [old_box], [], [], frame, frame, None, NoMatch(),
                   allow_motion_fallback=True)
    with patch('frame_processing.detect_pin_boxes_relaxed',
               side_effect=AssertionError('searched unchanged frame')):
        process_frame(job, lambda _index: pytest.fail('decoded cached frame'))


def test_panning_search_finds_first_pin_without_a_previous_track():
    previous = np.full((240, 360, 3), 125, dtype=np.uint8)
    current = previous.copy()
    cv2.circle(current, (300, 120), 34, (137, 137, 137), -1)
    cv2.circle(current, (300, 120), 34, (113, 113, 113), 2)

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [], [], [], current, previous, None, NoMatch(),
                   allow_motion_fallback=True)
    result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))
    assert result.primary_count == 0 and result.relaxed_count > 0
    assert any(center_inside(box.coords, (300, 120)) for box in result.boxes)


@pytest.mark.gui
def test_playback_draws_new_candidate_during_pan(tmp_path):
    app = wx.App.Get() or wx.App(False)
    previous = np.full((240, 360, 3), 125, dtype=np.uint8)
    current = previous.copy()
    for frame, x in ((previous, 95), (current, 65)):
        cv2.rectangle(frame, (x, 65), (x + 55, 125), (25, 35, 190), -1)
        cv2.rectangle(frame, (x, 65), (x + 55, 125), (10, 10, 10), 3)
    cv2.circle(current, (300, 120), 34, (137, 137, 137), -1)
    cv2.circle(current, (300, 120), 34, (113, 113, 113), 2)
    window = VideoScrubber(None, 'Pan detection', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                        for frame in (previous, current)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window._ScrubberFrame__frame_boxes[0] = [
            BoxData(detect_pin_boxes(previous)[0], ['Known'], 'automatic',
                    track_id='known', review_state='confirmed',
                    identity_confirmed=True)]
        window._ScrubberFrame__detected_frames.add(0)
        window.Show()
        wx.Yield()
        window.on_play(None)
        window._on_play_timer(None)
        deadline = monotonic() + 5
        while window.current_index != 1 and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert window.current_index == 1
        assert any(center_inside(box.coords, (300, 120))
                   for box in window._ScrubberFrame__frame_boxes[1])
    finally:
        window.pause_playback()
        window.Destroy()
        wx.Yield()


def test_existing_annotation_wins_over_a_tracking_proposal():
    manual = BoxData((43, 63, 60, 65), ['Corrected'], 'user')
    tracked = BoxData((48, 66, 60, 65), ['Old'], 'automatic')

    boxes = add_unique_boxes([manual], [tracked])

    assert boxes == [manual]


def test_candidate_inside_most_of_another_area_is_not_added():
    broad = BoxData((0, 0, 100, 100), [], 'automatic')
    mostly_inside = BoxData((70, 50, 50, 50), [], 'automatic')
    slightly_overlapping = BoxData((95, 0, 50, 50), [], 'automatic')

    assert add_unique_boxes([broad], [mostly_inside]) == [broad]
    assert add_unique_boxes([broad], [slightly_overlapping]) == [broad, slightly_overlapping]


def test_same_pin_overlapping_boxes_merge_without_losing_direct_review():
    reviewed = BoxData((40, 40, 45, 45), ['Mario'], 'user',
                       track_id='reviewed-track', pin_id='pinpanion:7',
                       review_state='confirmed', feedback_id='review-7',
                       feedback_label='positive', identity_confirmed=True)
    duplicate = BoxData((43, 42, 44, 46), ['Mario'], 'automatic',
                        track_id='other-track', pin_id='pinpanion:7')

    kept, removed = consolidate_overlapping_boxes([duplicate, reviewed])

    assert kept == [reviewed]
    assert removed == [duplicate]
    assert reviewed.feedback_id == 'review-7'
    assert reviewed.coords == (40, 40, 45, 45)


def test_conflicting_confirmed_pins_are_not_merged():
    first = BoxData((40, 40, 45, 45), ['Mario'], 'user',
                    pin_id='pinpanion:7', review_state='confirmed',
                    identity_confirmed=True)
    second = BoxData((43, 42, 44, 46), ['Luigi'], 'user',
                     pin_id='pinpanion:8', review_state='confirmed',
                     identity_confirmed=True)

    assert consolidate_overlapping_boxes([first, second]) == ([first, second], [])


def test_small_overlap_and_manual_name_survive_candidate_cleanup():
    named = BoxData((40, 40, 40, 40), ['Hand entered'], 'automatic')
    nearby = BoxData((78, 40, 40, 40), [], 'automatic')
    overlap = BoxData((42, 42, 40, 40), ['Catalog guess'], 'automatic',
                      pin_id='pinpanion:7', match_confidence=0.95)

    kept, removed = consolidate_overlapping_boxes([named, nearby, overlap])

    assert kept == [named, nearby]
    assert removed == [overlap]
    assert named.tags == ['Hand entered']


def test_wrong_identity_review_is_not_merged_into_rejected_catalog_id():
    reviewed = BoxData(
        (40, 40, 40, 40), [], 'automatic',
        identity_rejections=(IdentityRejection(
            'wrong-7', 'pinpanion:7', ('Mario',), (40, 40, 40, 40)),))
    suggestion = BoxData((42, 42, 40, 40), ['Mario'], 'automatic',
                         pin_id='pinpanion:7')

    assert consolidate_overlapping_boxes([reviewed, suggestion]) == (
        [reviewed, suggestion], [])


def test_worker_keeps_reviewed_track_beside_unrelated_saved_proposal():
    image = np.random.default_rng(213).integers(0, 255, (140, 180, 3), dtype=np.uint8)
    source = BoxData((50, 45, 40, 40), [], 'user', review_state='confirmed')
    broad = BoxData((20, 15, 110, 110), [], 'automatic')

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [broad], [], image, image, None, NoMatch())
    with patch('frame_processing.detect_pin_boxes', return_value=[]):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert result.boxes[0] is broad and broad.coords == (20, 15, 110, 110)
    tracked = [box for box in result.boxes if box.track_id == source.track_id]
    assert len(tracked) == 1
    assert tracked[0].coords == source.coords
    assert tracked[0].review_state == 'inherited'


def test_tracking_copies_only_matched_tags():
    previous = make_frame(include_second=False)
    current = make_frame(first_x=50)
    old = BoxData((43, 63, 60, 65), ['Mario'], 'user')

    boxes = associate_frame_boxes(previous, current, [old], detect_pin_boxes(current))

    assert len(boxes) == 2
    tracked = next(box for box in boxes if center_inside(box.coords, (78, 98)))
    new = next(box for box in boxes if center_inside(box.coords, (245, 115)))
    assert tracked.tags == ['Mario']
    assert tracked.prev_box_id == old.id
    assert tracked.track_id == old.track_id
    assert new.tags == []
    assert new.prev_box_id is None
    assert new.track_id != old.track_id


def test_different_appearance_does_not_inherit_tag():
    previous = make_frame(include_second=False)
    current = np.full_like(previous, BACKGROUND)
    cv2.circle(current, (73, 95), 30, (210, 55, 30), -1)
    cv2.circle(current, (73, 95), 30, (5, 5, 5), 3)
    old = BoxData((43, 63, 60, 65), ['Mario'], 'user')

    boxes = associate_frame_boxes(previous, current, [old], detect_pin_boxes(current))

    assert len(boxes) == 1
    assert boxes[0].tags == []


def test_local_search_keeps_a_tag_when_proposal_is_missed():
    previous = make_frame(include_second=False)
    current = make_frame(first_x=50, include_second=False)
    old = BoxData((43, 63, 60, 65), ['Mario'], 'user')

    boxes = associate_frame_boxes(previous, current, [old], [])

    assert len(boxes) == 1
    assert center_inside(boxes[0].coords, (78, 98))
    assert boxes[0].tags == ['Mario']


def test_local_search_rejects_a_different_object():
    previous = make_frame(include_second=False)
    current = np.full_like(previous, BACKGROUND)
    cv2.circle(current, (73, 95), 30, (210, 55, 30), -1)
    old = BoxData((43, 63, 60, 65), ['Mario'], 'user')

    assert associate_frame_boxes(previous, current, [old], []) == []


def test_local_search_ignores_a_featureless_region():
    frame = np.full((240, 360, 3), BACKGROUND, dtype=np.uint8)
    old = BoxData((43, 63, 60, 65), ['Mario'], 'user')

    assert associate_frame_boxes(frame, frame.copy(), [old], []) == []


def test_local_search_replaces_a_broad_untagged_proposal():
    previous = make_frame(include_second=False)
    current = make_frame(first_x=50, include_second=False)
    old = BoxData((43, 63, 60, 65), ['Mario'], 'user')

    boxes = associate_frame_boxes(previous, current, [old], [(30, 40, 200, 170)])

    assert len(boxes) == 1
    assert boxes[0].tags == ['Mario']
    assert boxes[0].coords[2:] == old.coords[2:]


def edge_entry_frames():
    pattern = np.random.default_rng(932).integers(15, 235, (60, 60, 3), dtype=np.uint8)
    previous = np.full((160, 200, 3), BACKGROUND, dtype=np.uint8)
    current = previous.copy()
    previous[50:110, 185:200] = pattern[:, :15]
    current[50:110, 135:195] = pattern
    return previous, current


@pytest.mark.parametrize('rotation,partial_coords', [
    (0, (185, 50, 15, 60)),
    (1, (50, 0, 60, 15)),
    (2, (0, 50, 15, 60)),
    (3, (50, 185, 60, 15)),
])
@pytest.mark.parametrize('saved_target', [False, True])
@pytest.mark.parametrize('has_proposal', [False, True])
def test_play_tracking_grows_unconfirmed_edge_fragment(
        rotation, partial_coords, saved_target, has_proposal):
    previous, current = edge_entry_frames()
    previous = np.ascontiguousarray(np.rot90(previous, rotation))
    current = np.ascontiguousarray(np.rot90(current, rotation))
    source = BoxData(partial_coords, [], 'automatic', track_id='edge')
    saved = BoxData(partial_coords, [], 'automatic') if saved_target else None

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [saved] if saved else [], [],
                   current, previous, None, NoMatch())
    with patch('frame_processing.detect_pin_boxes',
               return_value=[partial_coords] if has_proposal else []):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    tracked = [box for box in result.boxes if box.track_id == 'edge']
    assert len(tracked) == 1
    if saved_target:
        assert tracked[0] is saved
    x, y, width, height = tracked[0].coords
    assert width >= 55 and height >= 55
    assert 0 <= x < x + width <= current.shape[1]
    assert 0 <= y < y + height <= current.shape[0]


def changing_size_frames():
    previous = np.full((160, 200, 3), BACKGROUND, dtype=np.uint8)
    current = previous.copy()
    cv2.rectangle(previous, (60, 45), (129, 114), (215, 45, 60), -1)
    cv2.line(previous, (67, 50), (118, 106), (15, 220, 220), 3)
    cv2.rectangle(current, (75, 60), (114, 99), (215, 45, 60), -1)
    cv2.line(current, (79, 65), (109, 94), (15, 220, 220), 3)
    return previous, current


@pytest.mark.parametrize('has_proposal', [False, True])
def test_play_tracking_tightens_box_as_object_shrinks(has_proposal):
    previous, current = changing_size_frames()
    source = BoxData((60, 45, 70, 70), [], 'automatic',
                     track_id='shrinking', review_state='confirmed')

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [], [], current, previous, None, NoMatch())
    proposals = [(60, 45, 70, 70)] if has_proposal else []
    with patch('frame_processing.detect_pin_boxes', return_value=proposals):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    tracked = [box for box in result.boxes if box.track_id == 'shrinking']
    assert len(tracked) == 1
    assert tracked[0].coords == (75, 60, 40, 40)


@pytest.mark.parametrize('saved_target', [False, True])
@pytest.mark.parametrize('has_proposal', [False, True])
def test_play_tracking_expands_box_as_object_grows(saved_target, has_proposal):
    large, small = changing_size_frames()
    source = BoxData((75, 60, 40, 40), [], 'automatic',
                     track_id='growing', review_state='confirmed')
    saved = BoxData(source.coords, [], 'automatic') if saved_target else None

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [saved] if saved else [], [],
                   large, small, None, NoMatch())
    with patch('frame_processing.detect_pin_boxes',
               return_value=[source.coords] if has_proposal else []):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    tracked = [box for box in result.boxes if box.track_id == 'growing']
    assert len(tracked) == 1
    if saved_target:
        assert tracked[0] is saved
    assert tracked[0].coords == (60, 45, 70, 70)


def test_play_updates_saved_detector_box_as_object_shrinks():
    previous, current = changing_size_frames()
    source = BoxData((60, 45, 70, 70), [], 'automatic',
                     track_id='shrinking', review_state='confirmed')
    saved = BoxData((60, 45, 70, 70), [], 'automatic')

    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [saved], [], current, previous, None, NoMatch())
    with patch('frame_processing.detect_pin_boxes', return_value=[saved.coords]):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert result.boxes[0] is saved
    assert saved.coords == (75, 60, 40, 40)
    assert saved.track_id == source.track_id
    assert saved.bounds_adjustment is None


def test_play_scales_user_crop_margin_when_object_shrinks():
    previous, current = changing_size_frames()
    source = BoxData((60, 45, 70, 70), [], 'automatic',
                     track_id='shrinking', review_state='confirmed',
                     bounds_adjustment=(-5, -5, 10, 10), bounds_direct=True)
    tracked = associate_frame_boxes(previous, current, [source], [(60, 45, 70, 70)])

    assert len(tracked) == 1
    assert tracked[0].track_id == source.track_id
    x, y, width, height = tracked[0].coords
    assert x <= 75 and y <= 60
    assert x + width >= 115 and y + height >= 100
    assert width < 55 and height < 55


@pytest.mark.gui
def test_play_commits_tighter_bounds_on_existing_box():
    app = wx.App.Get() or wx.App(False)
    previous, current = changing_size_frames()
    window = VideoScrubber(None, 'Playback crop fit', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                        for image in (previous, current)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        source = BoxData((60, 45, 70, 70), [], 'automatic',
                         track_id='shrinking', review_state='confirmed')
        saved = BoxData((60, 45, 70, 70), [], 'automatic')
        window._ScrubberFrame__frame_boxes.update({0: [source], 1: [saved]})
        window._ScrubberFrame__detected_frames.add(0)
        window.display_image(reconcile=False)
        with patch('frame_processing.detect_pin_boxes', return_value=[saved.coords]):
            window.on_play(None)
            window._on_play_timer(None)
            deadline = monotonic() + 5
            while window.current_index != 1 and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert window.current_index == 1
        assert window._ScrubberFrame__frame_boxes[1][0] is saved
        assert saved.coords == (75, 60, 40, 40)
        assert saved.track_id == source.track_id
    finally:
        window.Destroy()
        wx.Yield()


def test_detector_keeps_a_narrow_pin_fragment_touching_frame_edge():
    previous, _current = edge_entry_frames()

    proposals = detect_pin_boxes(previous)

    assert any(center_inside(box, (192, 80)) and box[0] + box[2] == 200
               for box in proposals)


def corrected_bounds_frames():
    previous = np.full((140, 180, 3), 125, dtype=np.uint8)
    current = previous.copy()
    for frame, center in ((previous, (70, 70)), (current, (75, 70))):
        cv2.circle(frame, center, 20, (220, 40, 35), -1)
        cv2.circle(frame, center, 12, (30, 130, 220), -1)
        cv2.line(frame, (center[0] - 12, 62), (center[0] + 12, 78),
                 (255, 255, 255), 2)
    return previous, current


def test_user_corrected_bounds_follow_adjacent_detector_motion():
    previous, current = corrected_bounds_frames()
    source = BoxData((45, 45, 50, 50), [], 'automatic',
                     track_id='corrected')
    source.bounds_adjustment = (-5, -5, 10, 10)

    tracked = associate_frame_boxes(previous, current, [source], [(55, 50, 40, 40)])

    assert len(tracked) == 1
    assert tracked[0].track_id == source.track_id
    assert tracked[0].coords == (50, 45, 50, 50)
    assert tracked[0].bounds_adjustment == source.bounds_adjustment


def test_corrected_bounds_replace_unreviewed_saved_detector_box_on_worker():
    previous, current = corrected_bounds_frames()
    source = BoxData((45, 45, 50, 50), [], 'automatic', track_id='corrected',
                     bounds_adjustment=(-5, -5, 10, 10))
    saved = BoxData((55, 50, 40, 40), [], 'automatic')
    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [saved], [], current, previous, None, NoMatch())
    with patch('frame_processing.detect_pin_boxes', return_value=[saved.coords]):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert len(result.boxes) == 1
    assert result.boxes[0].coords == (50, 45, 50, 50)
    assert result.boxes[0].bounds_adjustment == (-5, -5, 10, 10)


def test_adjacent_tracking_preserves_destination_users_own_bounds():
    previous, current = corrected_bounds_frames()
    source = BoxData((45, 45, 50, 50), [], 'automatic',
                     bounds_adjustment=(-5, -5, 10, 10))
    saved = BoxData((52, 47, 46, 46), [], 'automatic',
                    bounds_adjustment=(-3, -3, 6, 6), bounds_direct=True)
    class NoMatch:
        def match(self, _crop):
            return None

    job = FrameJob(1, 0, [source], [saved], [], current, previous, None, NoMatch())
    with patch('frame_processing.detect_pin_boxes', return_value=[(55, 50, 40, 40)]):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))

    assert result.boxes[0].coords == saved.coords
    assert result.boxes[0].bounds_adjustment == saved.bounds_adjustment


def test_adjacent_correction_can_be_revised_and_withdrawn_from_saved_box():
    previous, current = corrected_bounds_frames()
    source = BoxData((45, 45, 50, 50), [], 'automatic', track_id='corrected',
                     bounds_adjustment=(-5, -5, 10, 10), bounds_direct=True)
    saved = BoxData((50, 45, 50, 50), [], 'automatic', track_id='corrected',
                    bounds_adjustment=(-5, -5, 10, 10))
    frames = {0: [source], 1: [saved]}
    read_frame = lambda index: (previous, current)[index]

    source.coords = (42, 42, 56, 56)
    source.bounds_adjustment = (-8, -8, 16, 16)
    assert reconcile_saved_frames(1, 2, frames, read_frame, reference_index=0)
    assert saved.coords == (47, 42, 56, 56)
    assert saved.bounds_adjustment == source.bounds_adjustment

    source.coords = (50, 50, 40, 40)
    source.bounds_adjustment = None
    source.bounds_direct = False
    assert reconcile_saved_frames(1, 2, frames, read_frame, reference_index=0)
    assert saved.coords == (55, 50, 40, 40)
    assert saved.bounds_adjustment is None


def test_adjacent_correction_never_overwrites_direct_destination_edit():
    previous, current = corrected_bounds_frames()
    source = BoxData((42, 42, 56, 56), [], 'automatic',
                     bounds_adjustment=(-8, -8, 16, 16), bounds_direct=True)
    saved = BoxData((52, 47, 46, 46), [], 'automatic',
                    bounds_adjustment=(-3, -3, 6, 6), bounds_direct=True)

    assert not inherit_corrected_bounds(previous, current, source, saved)
    assert saved.coords == (52, 47, 46, 46)
    assert saved.bounds_adjustment == (-3, -3, 6, 6)


@pytest.mark.gui
@pytest.mark.parametrize('verified_identity', [False, True])
def test_next_frame_commits_corrected_bounds_to_saved_box_object(verified_identity):
    app = wx.App.Get() or wx.App(False)
    previous, current = corrected_bounds_frames()
    window = VideoScrubber(None, 'Corrected adjacent bounds', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                        for image in (previous, current)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        source = BoxData((45, 45, 50, 50), [], 'automatic',
                         track_id='corrected', bounds_adjustment=(-5, -5, 10, 10),
                         pin_id='pin-7' if verified_identity else None,
                         review_state='confirmed' if verified_identity else 'unconfirmed',
                         identity_confirmed=verified_identity)
        saved = BoxData((55, 50, 40, 40), [], 'automatic',
                        pin_id='pin-7' if verified_identity else None,
                        review_state='inherited' if verified_identity else 'unconfirmed',
                        identity_confirmed=verified_identity)
        window._ScrubberFrame__frame_boxes[0] = [source]
        window._ScrubberFrame__frame_boxes[1] = [saved]
        window.display_image(reconcile=False)
        with patch('frame_processing.detect_pin_boxes', return_value=[saved.coords]):
            window.on_next_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert window.current_index == 1
        assert window._ScrubberFrame__frame_boxes[1][0] is saved
        assert saved.coords == (50, 45, 50, 50)
        assert saved.bounds_adjustment == source.bounds_adjustment
        assert source.coords == (45, 45, 50, 50)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_revisiting_processed_frame_reconciles_corrected_bounds():
    app = wx.App.Get() or wx.App(False)
    previous, current = corrected_bounds_frames()
    window = VideoScrubber(None, 'Revisit corrected bounds', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                                        for image in (previous, current)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        source = BoxData((45, 45, 50, 50), [], 'automatic',
                         track_id='corrected', bounds_adjustment=(-5, -5, 10, 10))
        saved = BoxData((55, 50, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [source]
        window._ScrubberFrame__frame_boxes[1] = [saved]
        window._ScrubberFrame__detected_frames.update((0, 1))
        window.display_image(reconcile=False)

        window.on_next_async(None)
        assert window._pending_frame_job is None
        future = window._pending_reconcile
        assert future is not None
        future.result(timeout=5)
        deadline = monotonic() + 5
        while window._pending_reconcile is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)

        assert window._ScrubberFrame__frame_boxes[1][0] is saved
        assert saved.coords == (50, 45, 50, 50)
        assert saved.bounds_adjustment == source.bounds_adjustment
    finally:
        window.Destroy()
        wx.Yield()


def test_confirmed_edge_fragment_tracks_larger_detector_box():
    previous, current = edge_entry_frames()
    partial = BoxData((185, 50, 15, 60), ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)
    full = (135, 50, 60, 60)

    tracked = associate_frame_boxes(previous, current, [partial], [full])

    assert len(tracked) == 1
    assert tracked[0].coords == full
    assert tracked[0].track_id == partial.track_id
    assert tracked[0].identity_confirmed
    assert tracked[0].review_state == 'inherited'


def test_saved_edge_fragment_links_to_expanded_box_in_both_directions():
    previous, current = edge_entry_frames()
    partial = BoxData((185, 50, 15, 60), ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)
    full = BoxData((135, 50, 60, 60), [], 'automatic')

    assert match_existing_candidates(previous, current, [partial], [full]) == [
        (partial, full)]
    assert match_existing_candidates(current, previous, [full], [partial]) == []
    full.review_state = 'inherited'
    full.pin_id = 'pin-7'
    full.tags = ['Mario']
    full.identity_confirmed = True
    assert match_existing_candidates(current, previous, [full], [partial]) == [
        (full, partial)]


def test_edge_fragment_can_expand_after_local_search_when_proposal_is_missing():
    previous, current = edge_entry_frames()
    partial = BoxData((185, 50, 15, 60), ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)

    tracked = associate_frame_boxes(previous, current, [partial], [])

    assert len(tracked) == 1
    x, y, width, height = tracked[0].coords
    assert x <= 140 and x + width >= 190
    assert width >= 50 and height >= 55
    assert 0 <= x < x + width <= 200
    assert tracked[0].track_id == partial.track_id


def test_left_edge_fragment_expands_even_if_new_box_is_still_near_edge():
    right_previous, right_current = edge_entry_frames()
    previous = np.ascontiguousarray(np.fliplr(right_previous))
    current = np.ascontiguousarray(np.fliplr(right_current))
    partial = BoxData((0, 50, 15, 60), ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)

    for proposals in ([(5, 50, 60, 60)], []):
        tracked = associate_frame_boxes(previous, current, [partial], proposals)
        assert len(tracked) == 1
        x, _y, width, _height = tracked[0].coords
        assert x <= 10 and x + width >= 60
        assert tracked[0].track_id == partial.track_id


def test_near_edge_small_annotation_can_expand_without_a_detector_proposal():
    pattern = np.random.default_rng(934).integers(15, 235, (60, 60, 3), dtype=np.uint8)
    previous = np.full((160, 200, 3), BACKGROUND, dtype=np.uint8)
    current = previous.copy()
    previous[50:110, 5:20] = pattern[:, :15]
    current[50:110, 5:65] = pattern
    partial = BoxData((5, 50, 15, 60), ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)

    tracked = associate_frame_boxes(previous, current, [partial], [])

    assert len(tracked) == 1
    assert tracked[0].coords[0] <= 10
    assert tracked[0].coords[2] >= 50
    assert tracked[0].track_id == partial.track_id


@pytest.mark.parametrize('rotation,partial_coords,full_coords', [
    (1, (50, 0, 60, 15), (50, 5, 60, 60)),
    (3, (50, 185, 60, 15), (50, 135, 60, 60)),
])
def test_top_and_bottom_fragments_track_expanded_boxes(
        rotation, partial_coords, full_coords):
    right_previous, right_current = edge_entry_frames()
    previous = np.ascontiguousarray(np.rot90(right_previous, rotation))
    current = np.ascontiguousarray(np.rot90(right_current, rotation))
    partial = BoxData(partial_coords, ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)

    tracked = associate_frame_boxes(previous, current, [partial], [full_coords])

    assert len(tracked) == 1
    assert tracked[0].coords == full_coords
    assert tracked[0].track_id == partial.track_id


def test_edge_expansion_does_not_copy_identity_to_different_appearance():
    previous, current = edge_entry_frames()
    current[50:110, 135:195] = np.random.default_rng(933).integers(
        15, 235, (60, 60, 3), dtype=np.uint8)
    partial = BoxData((185, 50, 15, 60), ['Mario'], 'automatic',
                      pin_id='pin-7', review_state='confirmed', identity_confirmed=True)

    tracked = associate_frame_boxes(previous, current, [partial], [(135, 50, 60, 60)])

    assert len(tracked) == 1
    assert tracked[0].track_id != partial.track_id
    assert not tracked[0].identity_confirmed


def test_next_frame_keeps_confirmed_edge_pin_and_expands_its_box_in_gui():
    app = wx.App.Get() or wx.App(False)
    previous, current = edge_entry_frames()
    window = VideoScrubber(None, 'Pin entering frame', '',
                           image_array=[cv2.cvtColor(previous, cv2.COLOR_RGB2BGR),
                                        cv2.cvtColor(current, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        partial = BoxData((185, 50, 15, 60), ['Mario'], 'automatic',
                          pin_id='pin-7', review_state='confirmed',
                          identity_confirmed=True)
        window._ScrubberFrame__frame_boxes[0] = [partial]
        window._ScrubberFrame__detected_frames.add(0)
        window.display_image(reconcile=False)
        with patch('frame_processing.detect_pin_boxes', return_value=[(135, 50, 60, 60)]):
            window.on_next_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        boxes = window._ScrubberFrame__frame_boxes[1]
        assert len(boxes) == 1
        assert boxes[0].coords == (135, 50, 60, 60)
        assert boxes[0].track_id == partial.track_id
        assert boxes[0].identity_confirmed and boxes[0].review_state == 'inherited'
        assert partial.coords == (185, 50, 15, 60)
    finally:
        window.Destroy()
        wx.Yield()


def test_track_id_survives_json_round_trip_and_old_json_loads():
    box = BoxData((43, 63, 60, 65), ['Mario'], 'user')
    output = StringIO()
    save_boxes_to_stream(output, {3: [box]})
    output.seek(0)

    loaded = load_boxes_from_stream(output)[3][0]
    legacy = load_boxes_from_stream(StringIO('{"3": [{"coords": [1, 2, 3, 4], "tags": ["old"]}]}'))[3][0]

    assert loaded.track_id == box.track_id
    assert legacy.track_id
    assert legacy.coords == (1, 2, 3, 4)


def test_next_frame_detects_and_tracks_in_gui():
    app = wx.App.Get() or wx.App(False)
    previous = make_frame(include_second=False)
    current = make_frame(first_x=50)
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in (previous, current)]
    window = VideoScrubber(None, 'Detection test', '', image_array=frames, load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        old = BoxData((43, 63, 60, 65), ['Mario'], 'user')
        window._ScrubberFrame__frame_boxes[0] = [old]
        window.display_image()

        window.on_next(None)

        boxes = window._ScrubberFrame__frame_boxes[1]
        assert window.current_index == 1
        assert len(boxes) == 2
        assert next(box for box in boxes if center_inside(box.coords, (78, 98))).tags == ['Mario']
        assert next(box for box in boxes if center_inside(box.coords, (245, 115))).tags == []
        assert len(window._ScrubberFrame__tag_panel.boxes) == len(boxes)
    finally:
        window.Destroy()
        wx.Yield()


def test_detect_button_keeps_manual_annotation():
    app = wx.App.Get() or wx.App(False)
    frame = make_frame()
    window = VideoScrubber(None, 'Detection test', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        manual = BoxData((43, 63, 60, 65), ['Mario'], 'user')
        window._ScrubberFrame__frame_boxes[0] = [manual]
        button = window._ScrubberFrame__button_panel.process_btn
        click = wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId())
        click.SetEventObject(button)

        button.ProcessEvent(click)
        window._pending_frame_job.result(timeout=3)
        from time import monotonic, sleep
        deadline = monotonic() + 3
        while 0 not in window._ScrubberFrame__detected_frames and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert 0 in window._ScrubberFrame__detected_frames

        boxes = window._ScrubberFrame__frame_boxes[0]
        assert len(boxes) == 2
        assert boxes[0] is manual
        assert boxes[1].tags == []
    finally:
        window.Destroy()
        wx.Yield()
