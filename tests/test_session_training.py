from time import monotonic, sleep
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData, IdentityRejection
from controls.BoxTagEditPanel import BoxTagPanelEdit
from detection_feedback import FeedbackStore
from session_training import (build_training_examples, reviewed_observations,
                              plan_training_delta)
from videoscrubber import VideoScrubber


def textured_frame() -> np.ndarray:
    return np.random.default_rng(81).integers(0, 255, (90, 120, 3), dtype=np.uint8)


def test_training_uses_direct_reviews_and_preserves_identity_labels():
    direct = BoxData((10, 10, 35, 35), ['Mario'], 'automatic', track_id='pin',
                     pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=True)
    inherited = BoxData((10, 10, 35, 35), ['Mario'], 'automatic', track_id='pin',
                        pin_id='pinpanion:7', review_state='inherited',
                        identity_confirmed=True)
    rejected = BoxData((60, 10, 35, 35), ['Wrong suggestion'], 'automatic',
                       track_id='noise', review_state='rejected',
                       feedback_id='negative-id', feedback_label='negative')
    propagated_rejection = BoxData((60, 10, 35, 35), [], 'automatic',
                                   track_id='noise', review_state='rejected')
    observations = reviewed_observations(
        {0: [direct, rejected], 1: [inherited, propagated_rejection]}, 'video.mp4')
    assert [(item.frame_index, item.label) for item in observations] == [
        (0, 'positive'), (0, 'negative')]
    updates = []
    examples, skipped = build_training_examples(
        observations, 'video.mp4', lambda _index: textured_frame(),
        lambda done, total, eta: updates.append((done, total, eta)))
    assert skipped == 0
    assert updates[0][:2] == (0, 2) and updates[-1][:2] == (2, 2)
    assert examples[0].pin_id == 'pinpanion:7'
    assert examples[0].tags == ('Mario',)
    assert examples[1].pin_id is None and examples[1].tags == ()


def test_training_delta_skips_unchanged_and_detects_edits_and_deletions():
    original = BoxData((10, 10, 35, 35), ['Mario'], 'automatic',
                       track_id='first', pin_id='pinpanion:7',
                       review_state='confirmed', identity_confirmed=True,
                       feedback_id='first-id', feedback_label='positive')
    removed = BoxData((60, 10, 35, 35), [], 'automatic',
                      track_id='removed', review_state='rejected',
                      feedback_id='removed-id', feedback_label='negative')
    first, second = reviewed_observations({0: [original, removed]}, 'video.mp4')
    stored = {
        first.example_id: ('positive', 0, 'first', 'pinpanion:7',
                           (10, 10, 35, 35), ('Mario',)),
        second.example_id: ('negative', 0, 'removed', None,
                            (60, 10, 35, 35), ())}
    delta = plan_training_delta([first, second], stored)
    assert delta.pending == [] and delta.removed_ids == set()
    original.coords = (11, 10, 35, 35)
    added = BoxData((60, 10, 35, 35), ['Luigi'], 'automatic',
                    track_id='new', review_state='confirmed',
                    feedback_id='new-id', feedback_label='positive')
    observations = reviewed_observations({0: [original, added]}, 'video.mp4')
    delta = plan_training_delta(observations, stored)
    assert [item.example_id for item in delta.pending] == ['first-id', 'new-id']
    assert delta.new_count == 1 and delta.updated_count == 1
    assert delta.removed_ids == {'removed-id'}


def test_training_indexes_every_rejected_identity_once_across_tracked_frames():
    frame_boxes: dict[int, list[BoxData]] = {}
    for box_index in range(40):
        track_id = f'track-{box_index}'
        reviews = tuple(IdentityRejection(
            f'wrong-{box_index}-{candidate}', f'pin-{candidate}',
            (f'Pin {candidate}',), (10, 10, 35, 35), box_index)
            for candidate in range(15))
        source = BoxData((10, 10, 35, 35), [], 'automatic',
                         track_id=track_id, identity_rejections=reviews)
        propagated = BoxData((10, 10, 35, 35), [], 'automatic',
                             track_id=track_id, identity_rejections=reviews)
        frame_boxes[box_index] = [source]
        frame_boxes[box_index + 40] = [propagated]

    observations = reviewed_observations(frame_boxes, 'video.mp4')
    assert len(observations) == 600
    assert {item.label for item in observations} == {'wrong_identity'}
    assert len({item.example_id for item in observations}) == 600
    delta = plan_training_delta(observations, {})
    assert len(delta.pending) == 600

    frame = textured_frame()
    examples, skipped = build_training_examples(
        delta.pending, 'video.mp4', lambda _index: frame,
        lambda _done, _total, _eta: None)
    assert skipped == 0 and len(examples) == 600
    store = FeedbackStore()
    try:
        store.apply_source_delta('video.mp4', examples, set())
        assert len(store.identity_rejections_for_source('video.mp4')) == 600
        assert store.counts() == {'positive': 0, 'negative': 0}
        assert plan_training_delta(
            observations, store.training_metadata_for_source('video.mp4')).pending == []
        edited = [
            item if item.example_id != 'wrong-0-0' else
            type(item)(item.frame_index, item.coords, item.track_id,
                       item.label, item.pin_id, ('Corrected name',), item.example_id)
            for item in observations]
        changed = plan_training_delta(
            edited, store.training_metadata_for_source('video.mp4'))
        assert [item.example_id for item in changed.pending] == ['wrong-0-0']
        replacement, skipped = build_training_examples(
            changed.pending, 'video.mp4', lambda _index: frame,
            lambda _done, _total, _eta: None)
        assert skipped == 0
        store.apply_source_delta('video.mp4', replacement, set())
        assert store.training_metadata_for_source('video.mp4')['wrong-0-0'][-1] == (
            'Corrected name',)

        remaining = [item for item in edited if item.example_id != 'wrong-0-0']
        deleted = plan_training_delta(
            remaining, store.training_metadata_for_source('video.mp4'))
        assert deleted.pending == [] and deleted.removed_ids == {'wrong-0-0'}
        store.apply_source_delta('video.mp4', [], deleted.removed_ids)
        assert store.identity_rejection_count('video.mp4') == 599
    finally:
        store.close()


@pytest.mark.gui
def test_catalog_picker_sets_id_and_allows_free_text():
    app = wx.App.Get() or wx.App(False)
    frame = wx.Frame(None)
    box = BoxData((1, 1, 20, 20), [''], 'user')
    panel = BoxTagPanelEdit(frame, box, catalog_options=[('pinpanion:7', 'Mario')])
    try:
        frame.Show()
        wx.Yield()
        row = panel.get_or_create_label(0)
        picker = next(child for child in row.GetChildren()
                      if isinstance(child, wx.TextCtrl))
        row._open_catalog(wx.CommandEvent())
        assert row.visible_options == ['Mario']
        row.select_catalog_option(0)
        wx.Yield()
        assert box.tags == ['Mario'] and box.pin_id == 'pinpanion:7'
        picker.ChangeValue('Custom pin')
        row.fire_edited_event()
        wx.Yield()
        assert box.tags == ['Custom pin'] and box.pin_id is None
    finally:
        frame.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_training_button_indexes_session_and_reports_completion(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = textured_frame()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    window = VideoScrubber(None, 'Training', '', image_array=[bgr, bgr, bgr],
                           load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window.box_data_filename = str(tmp_path / 'session.json')
        window._ScrubberFrame__frame_boxes[0] = [
            BoxData((10, 10, 35, 35), ['Mario'], 'automatic', track_id='pin',
                    pin_id='pinpanion:7', review_state='confirmed', identity_confirmed=True),
            BoxData((60, 10, 35, 35), [], 'automatic', track_id='noise',
                    review_state='rejected', feedback_id='negative-id',
                    feedback_label='negative')]
        window._ScrubberFrame__frame_boxes[1] = [
            BoxData((10, 10, 35, 35), ['Mario'], 'automatic', track_id='pin',
                    review_state='inherited', identity_confirmed=True)]
        window.Show()
        wx.Yield()
        panel = window._ScrubberFrame__tag_panel
        panel.training_btn.ProcessEvent(wx.CommandEvent(
            wx.EVT_BUTTON.typeId, panel.training_btn.GetId()))
        deadline = monotonic() + 5
        while window._training_future is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        wx.Yield()
        assert window._training_future is None
        assert panel.training_gauge.GetValue() == 100
        assert '2 examples indexed' in panel.training_status.GetLabel()
        assert window._feedback_model.sample_counts == {'positive': 1, 'negative': 1}
        assert window._feedback_store.examples()[0].tags == ('Mario',)
        assert (tmp_path / 'session.json').is_file()
        revision = window._feedback_model.version
        with patch.object(window, 'read_frame_for_processing',
                          side_effect=AssertionError('decoded unchanged frame')):
            panel.training_btn.ProcessEvent(wx.CommandEvent(
                wx.EVT_BUTTON.typeId, panel.training_btn.GetId()))
            wx.Yield()
        assert window._training_future is None
        assert window._feedback_model.version == revision
        assert 'Up to date' in panel.training_status.GetLabel()
        assert len(window._feedback_store.examples()) == 2

        new = BoxData((60, 10, 35, 35), ['Luigi'], 'automatic', track_id='new',
                      review_state='confirmed')
        window._ScrubberFrame__frame_boxes[2] = [new]
        with patch.object(window, 'read_frame_for_processing',
                          wraps=window.read_frame_for_processing) as reader:
            panel.training_btn.ProcessEvent(wx.CommandEvent(
                wx.EVT_BUTTON.typeId, panel.training_btn.GetId()))
            deadline = monotonic() + 5
            while window._training_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            wx.Yield()
            assert [call.args[0] for call in reader.call_args_list] == [2]
        assert len(window._feedback_store.examples()) == 3
        assert '1 added, 0 updated' in panel.training_status.GetLabel()

        original = window._ScrubberFrame__frame_boxes[0][0]
        original.coords = (11, 10, 35, 35)
        with patch.object(window, 'read_frame_for_processing',
                          wraps=window.read_frame_for_processing) as reader:
            panel.training_btn.ProcessEvent(wx.CommandEvent(
                wx.EVT_BUTTON.typeId, panel.training_btn.GetId()))
            deadline = monotonic() + 5
            while window._training_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            wx.Yield()
            assert [call.args[0] for call in reader.call_args_list] == [0]
        assert '0 added, 1 updated' in panel.training_status.GetLabel()

        window._ScrubberFrame__frame_boxes[0].pop(1)
        with patch.object(window, 'read_frame_for_processing',
                          side_effect=AssertionError('decoded for deletion')):
            panel.training_btn.ProcessEvent(wx.CommandEvent(
                wx.EVT_BUTTON.typeId, panel.training_btn.GetId()))
        assert window._training_future is None
        assert len(window._feedback_store.examples()) == 2
        assert '1 removed' in panel.training_status.GetLabel()
    finally:
        window.Destroy()
        wx.Yield()
