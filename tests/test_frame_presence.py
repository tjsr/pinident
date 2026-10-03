import json
from io import StringIO
from time import monotonic, sleep
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from frame_presence import FramePresence, checkbox_values, marked_pin_count
from pin_catalog import PinMatch
from pin_detection import detect_pin_boxes_relaxed
from scrubberframe import load_boxes_from_stream, load_presence_from_file, save_boxes_to_stream
from videoscrubber import VideoScrubber


def test_presence_counts_reviewed_or_identified_pins_and_round_trips(tmp_path):
    boxes = [
        BoxData((2, 2, 20, 20), ['unknown-1'], 'user'),
        BoxData((30, 2, 20, 20), [], 'automatic'),
        BoxData((58, 2, 20, 20), [], 'automatic', review_state='confirmed'),
        BoxData((86, 2, 20, 20), ['Mario'], 'automatic', pin_id='pinpanion:1'),
        BoxData((114, 2, 20, 20), [], 'automatic', review_state='rejected'),
    ]
    assert marked_pin_count(boxes) == 2
    assert checkbox_values(FramePresence(), boxes) == (True, True)
    duplicate = BoxData((140, 2, 20, 20), ['Mario'], 'automatic',
                        track_id=boxes[2].track_id)
    assert marked_pin_count([boxes[2], duplicate]) == 1

    stream = StringIO()
    save_boxes_to_stream(stream, {0: [], 1: boxes},
                         {0: FramePresence(pin_asserted=True, set_asserted=True)})
    data = json.loads(stream.getvalue())
    assert data['_frame_presence']['0'] == {'contains_pin': True, 'contains_set': True}
    assert len(load_boxes_from_stream(StringIO(stream.getvalue()))[1]) == 5
    path = tmp_path / 'annotations.json'
    path.write_text(stream.getvalue(), encoding='utf-8')
    assert load_presence_from_file(path) == {0: FramePresence(True, True)}


def test_relaxed_detector_finds_low_contrast_round_region():
    image = np.full((160, 200, 3), 128, dtype=np.uint8)
    cv2.circle(image, (80, 70), 27, (140, 140, 140), thickness=-1)
    boxes = detect_pin_boxes_relaxed(image)
    assert any(x <= 80 < x + w and y <= 70 < y + h for x, y, w, h in boxes)


@pytest.mark.gui
def test_checkbox_hints_retry_detection_and_follow_review_removal_undo():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(91).integers(0, 255, (120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Frame presence', '', image_array=[image] * 2,
                           load_catalog=False)
    first = (10, 15, 35, 35)
    second = (100, 15, 35, 35)

    def wait_for_job():
        future = window._pending_frame_job
        assert future is not None
        future.result(timeout=3)
        deadline = monotonic() + 3
        while window._pending_frame_job is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert window._pending_frame_job is None

    def click(checkbox):
        checkbox.SetValue(True)
        event = wx.CommandEvent(wx.EVT_CHECKBOX.typeId, checkbox.GetId())
        event.SetEventObject(checkbox)
        checkbox.ProcessEvent(event)

    try:
        window.Show()
        wx.Yield()
        panel = window._ScrubberFrame__tag_panel
        assert not panel.pin_cb.GetValue() and not panel.set_cb.GetValue()

        with patch('frame_processing.detect_pin_boxes', return_value=[]), \
             patch('frame_processing.detect_pin_boxes_relaxed',
                   side_effect=[[first], [first, second]]) as fallback:
            click(panel.pin_cb)
            wait_for_job()
            assert fallback.call_count == 1
            assert panel.pin_cb.GetValue() and not panel.set_cb.GetValue()
            candidate = window._ScrubberFrame__frame_boxes[0][0]
            assert candidate.coords == first and candidate.review_state == 'unconfirmed'

            window.confirm_box(candidate)
            assert panel.pin_cb.GetValue() and not panel.set_cb.GetValue()

            click(panel.set_cb)
            wait_for_job()
            assert fallback.call_count == 2
            boxes = window._ScrubberFrame__frame_boxes[0]
            assert len(boxes) == 2
            assert panel.pin_cb.GetValue() and panel.set_cb.GetValue()

        second_box = next(box for box in boxes if box.coords == second)
        window.confirm_box(second_box)
        window.remove_box(candidate)
        assert panel.pin_cb.GetValue() and not panel.set_cb.GetValue()
        window.remove_box(second_box)
        assert not panel.pin_cb.GetValue() and not panel.set_cb.GetValue()

        window.undo_action()
        assert panel.pin_cb.GetValue() and not panel.set_cb.GetValue()
        window.undo_action()
        assert panel.pin_cb.GetValue() and panel.set_cb.GetValue()

        window.goto_frame(1)
        assert not panel.pin_cb.GetValue() and not panel.set_cb.GetValue()
        window.goto_frame(0)
        assert panel.pin_cb.GetValue() and panel.set_cb.GetValue()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_catalog_identifications_check_pin_and_set_without_user_assertion():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(92).integers(0, 255, (120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Auto presence', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        with patch('frame_processing.detect_pin_boxes', return_value=[
                (10, 15, 35, 35), (100, 15, 35, 35)]), \
             patch('pin_catalog.PinMatcher.match', return_value=PinMatch(
                 'pinpanion:1', 'Mario', 12)):
            window.on_detect_async(None)
            window._pending_frame_job.result(timeout=3)
            deadline = monotonic() + 3
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        panel = window._ScrubberFrame__tag_panel
        assert panel.pin_cb.GetValue() and panel.set_cb.GetValue()
        assert window._frame_presence == {}
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_contains_pin_can_recover_box_from_confirmed_adjacent_frame():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(93).integers(0, 255, (120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Adjacent presence', '', image_array=[image, image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        original = BoxData((25, 25, 40, 40), ['Mario'], 'automatic',
                           review_state='confirmed')
        window._ScrubberFrame__frame_boxes[0] = [original]
        window.goto_frame(1)
        checkbox = window._ScrubberFrame__tag_panel.pin_cb
        checkbox.SetValue(True)
        event = wx.CommandEvent(wx.EVT_CHECKBOX.typeId, checkbox.GetId())
        event.SetEventObject(checkbox)
        with patch('frame_processing.detect_pin_boxes', return_value=[]), \
             patch('frame_processing.detect_pin_boxes_relaxed', return_value=[]) as relaxed:
            checkbox.ProcessEvent(event)
            window._pending_frame_job.result(timeout=3)
            deadline = monotonic() + 3
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            assert relaxed.call_count == 0
        found = window._ScrubberFrame__frame_boxes[1]
        assert len(found) == 1
        assert found[0].track_id == original.track_id
        assert found[0].review_state == 'inherited'
        assert checkbox.GetValue()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_loaded_unmet_presence_remains_checked_and_needs_detection(tmp_path):
    app = wx.App.Get() or wx.App(False)
    annotations = tmp_path / 'presence.json'
    annotations.write_text(json.dumps({
        '0': [], '_frame_presence': {'0': {'contains_pin': True,
                                          'contains_set': False}},
    }), encoding='utf-8')
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Loaded presence', '', image_array=[image],
                           load_catalog=False)
    try:
        window.box_data_filename = str(annotations)
        window.load_box_data()
        window.Show()
        wx.Yield()
        panel = window._ScrubberFrame__tag_panel
        assert panel.pin_cb.GetValue() and not panel.set_cb.GetValue()
        assert 0 not in window._ScrubberFrame__detected_frames
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_loaded_empty_frame_is_detected_on_manual_navigation(tmp_path):
    app = wx.App.Get() or wx.App(False)
    annotations = tmp_path / 'empty-frames.json'
    annotations.write_text(json.dumps({'0': [], '1': []}), encoding='utf-8')
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Loaded empty frame', '', image_array=[image, image],
                           load_catalog=False)
    candidate = (25, 20, 40, 40)
    try:
        window.box_data_filename = str(annotations)
        window.load_box_data()
        assert 1 not in window._ScrubberFrame__detected_frames
        window.Show()
        wx.Yield()
        with patch('frame_processing.detect_pin_boxes', return_value=[]), \
             patch('frame_processing.detect_pin_boxes_relaxed',
                   return_value=[candidate]) as fallback:
            window.on_next_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=3)
            deadline = monotonic() + 3
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            assert fallback.call_count == 1
        assert [box.coords for box in window._ScrubberFrame__frame_boxes[1]] == [candidate]
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_unmet_presence_assertion_is_undoable():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Presence undo', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        checkbox = window._ScrubberFrame__tag_panel.pin_cb
        checkbox.SetValue(True)
        event = wx.CommandEvent(wx.EVT_CHECKBOX.typeId, checkbox.GetId())
        event.SetEventObject(checkbox)
        with patch('frame_processing.detect_pin_boxes', return_value=[]), \
             patch('frame_processing.detect_pin_boxes_relaxed', return_value=[]):
            checkbox.ProcessEvent(event)
            window._pending_frame_job.result(timeout=3)
            deadline = monotonic() + 3
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert checkbox.GetValue()
        window.undo_action()
        assert not checkbox.GetValue()
        window.redo_action()
        assert checkbox.GetValue()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_explicit_detect_uses_relaxed_search_after_empty_primary_pass():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(94).integers(0, 255, (120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Empty primary detection', '',
                           image_array=[image], load_catalog=False)
    candidate = (25, 25, 40, 40)
    try:
        window.Show()
        wx.Yield()
        with patch('frame_processing.detect_pin_boxes', return_value=[]), \
             patch('frame_processing.detect_pin_boxes_relaxed',
                   return_value=[candidate]) as relaxed:
            window.on_detect_async(None)
            window._pending_frame_job.result(timeout=3)
            deadline = monotonic() + 3
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            assert relaxed.call_count == 1
        boxes = window._ScrubberFrame__frame_boxes[0]
        assert [box.coords for box in boxes] == [candidate]
        assert 'primary 0, fallback 1' in window.GetStatusBar().GetStatusText(1)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('direction', ['next', 'previous'])
def test_manual_step_uses_same_fallback_as_detect_button(direction):
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Manual detection parity', '',
                           image_array=[image, image], load_catalog=False)
    candidate = (25, 25, 40, 40)

    def wait_for_job():
        future = window._pending_frame_job
        assert future is not None
        future.result(timeout=3)
        deadline = monotonic() + 3
        while window._pending_frame_job is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert window._pending_frame_job is None

    try:
        window.Show()
        wx.Yield()
        if direction == 'previous':
            window._ScrubberFrame__detected_frames.add(1)
            assert window.goto_frame(1)
        with patch('frame_processing.detect_pin_boxes', return_value=[]) as primary, \
             patch('frame_processing.detect_pin_boxes_relaxed',
                   return_value=[candidate]) as fallback:
            if direction == 'next':
                window.on_next_async(None)
            else:
                window.on_prev(None)
            wait_for_job()
            target = window.current_index
            assert [box.coords for box in window._ScrubberFrame__frame_boxes[target]] == [candidate]

            window.on_detect_async(None)
            wait_for_job()
            assert [box.coords for box in window._ScrubberFrame__frame_boxes[target]] == [candidate]
            assert primary.call_count == 2
            assert fallback.call_count == 1
    finally:
        window.Destroy()
        wx.Yield()
