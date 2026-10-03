import cv2
import numpy as np
import wx
from io import StringIO

from boxdata import BoxData
from pin_detection import (add_new_detections, add_unique_boxes, associate_frame_boxes,
                           detect_pin_boxes, detect_pin_boxes_relaxed, refine_region)
from scrubberframe import load_boxes_from_stream, save_boxes_to_stream
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


def test_detection_does_not_replace_a_manual_box():
    manual = BoxData((43, 63, 60, 65), ['Mario'], 'user')
    boxes = add_new_detections([manual], detect_pin_boxes(make_frame()))

    assert len(boxes) == 2
    assert boxes[0] is manual
    assert boxes[1].tags == []


def test_existing_annotation_wins_over_a_tracking_proposal():
    manual = BoxData((43, 63, 60, 65), ['Corrected'], 'user')
    tracked = BoxData((48, 66, 60, 65), ['Old'], 'automatic')

    boxes = add_unique_boxes([manual], [tracked])

    assert boxes == [manual]


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
