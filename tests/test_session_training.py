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


def test_training_uses_each_linked_pin_frame_and_preserves_identity_labels():
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
    unconfirmed = BoxData((5, 50, 15, 15), [], 'automatic',
                          track_id='guess', review_state='unconfirmed')
    orphan = BoxData((25, 50, 15, 15), [], 'automatic',
                     track_id='orphan', review_state='inherited')
    observations = reviewed_observations(
        {0: [direct, rejected],
         1: [inherited, propagated_rejection, unconfirmed, orphan]}, 'video.mp4')
    assert [(item.frame_index, item.label) for item in observations] == [
        (0, 'positive'), (0, 'negative'), (1, 'positive')]
    assert len({item.example_id for item in observations}) == 3
    updates = []
    examples, skipped = build_training_examples(
        observations, 'video.mp4',
        lambda index: np.random.default_rng(81 + index).integers(
            0, 255, (90, 120, 3), dtype=np.uint8),
        lambda done, total, eta: updates.append((done, total, eta)))
    assert skipped == 0
    assert updates[0][:2] == (0, 3) and updates[-1][:2] == (3, 3)
    assert examples[0].pin_id == 'pinpanion:7'
    assert examples[0].tags == ('Mario',)
    assert examples[1].pin_id is None and examples[1].tags == ()
    assert examples[2].frame_index == 1 and examples[2].label == 'positive'
    assert examples[2].crop_png != examples[0].crop_png


def test_linked_frame_training_delta_tracks_edits_and_removal():
    direct = BoxData((10, 10, 35, 35), [], 'automatic', track_id='pin',
                     review_state='confirmed', feedback_id='direct-1',
                     feedback_label='positive')
    linked = BoxData((11, 10, 35, 35), [], 'automatic', track_id='pin',
                     review_state='inherited')
    observations = reviewed_observations({0: [direct], 1: [linked]}, 'video.mp4')
    stored = {item.example_id: (item.label, item.frame_index, item.track_id,
                                item.pin_id, item.coords, item.tags)
              for item in observations}
    assert plan_training_delta(observations, stored).pending == []
    linked.coords = (12, 10, 35, 35)
    edited = reviewed_observations({0: [direct], 1: [linked]}, 'video.mp4')
    delta = plan_training_delta(edited, stored)
    assert [item.frame_index for item in delta.pending] == [1]
    assert delta.updated_count == 1 and delta.new_count == 0
    removed = plan_training_delta(reviewed_observations({0: [direct]}, 'video.mp4'), stored)
    assert removed.removed_ids == {observations[1].example_id}


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
        window.GetEventHandler().ProcessEvent(wx.CommandEvent(
            wx.EVT_MENU.typeId, window._training_add_item.GetId()))
        deadline = monotonic() + 5
        while window._training_future is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        wx.Yield()
        assert window._training_future is None
        assert panel.training_gauge.GetValue() == 100
        assert '3 examples indexed' in panel.training_status.GetLabel()
        assert '2 pin crops (linked frames: 1' in panel.training_status.GetLabel()
        assert window._feedback_model.sample_counts == {'positive': 2, 'negative': 1}
        assert window._feedback_store.examples()[0].tags == ('Mario',)
        assert (tmp_path / 'session.json').is_file()
        assert not (tmp_path / 'feedback.svm.yml').exists()
        window._seed_feedback_from_annotations()
        assert window._feedback_store.counts() == {'positive': 2, 'negative': 1}
        revision = window._feedback_model.version
        with patch.object(window, 'read_frame_for_processing',
                          side_effect=AssertionError('decoded unchanged frame')):
            window.GetEventHandler().ProcessEvent(wx.CommandEvent(
                wx.EVT_MENU.typeId, window._training_add_item.GetId()))
            wx.Yield()
        assert window._training_future is None
        assert window._feedback_model.version == revision
        assert 'Up to date' in panel.training_status.GetLabel()
        assert len(window._feedback_store.examples()) == 3

        new = BoxData((60, 10, 35, 35), ['Luigi'], 'automatic', track_id='new',
                      review_state='confirmed')
        window._ScrubberFrame__frame_boxes[2] = [new]
        with patch.object(window, 'read_frame_for_processing',
                          wraps=window.read_frame_for_processing) as reader:
            window.GetEventHandler().ProcessEvent(wx.CommandEvent(
                wx.EVT_MENU.typeId, window._training_add_item.GetId()))
            deadline = monotonic() + 5
            while window._training_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            wx.Yield()
            assert [call.args[0] for call in reader.call_args_list] == [2]
        assert len(window._feedback_store.examples()) == 4
        assert '1 added now, 0 updated' in panel.training_status.GetLabel()

        original = window._ScrubberFrame__frame_boxes[0][0]
        original.coords = (11, 10, 35, 35)
        with patch.object(window, 'read_frame_for_processing',
                          wraps=window.read_frame_for_processing) as reader:
            window.GetEventHandler().ProcessEvent(wx.CommandEvent(
                wx.EVT_MENU.typeId, window._training_add_item.GetId()))
            deadline = monotonic() + 5
            while window._training_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            wx.Yield()
            assert [call.args[0] for call in reader.call_args_list] == [0]
        assert '0 added now, 1 updated' in panel.training_status.GetLabel()

        window._ScrubberFrame__frame_boxes[0].pop(1)
        with patch.object(window, 'read_frame_for_processing',
                          side_effect=AssertionError('decoded for deletion')):
            window.GetEventHandler().ProcessEvent(wx.CommandEvent(
                wx.EVT_MENU.typeId, window._training_add_item.GetId()))
        assert window._training_future is None
        assert len(window._feedback_store.examples()) == 3
        assert '1 removed' in panel.training_status.GetLabel()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_drawn_unidentified_pins_are_reported_and_used_for_training(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = textured_frame()
    window = VideoScrubber(None, 'Unidentified pin training', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False,
                           catalog_path=str(tmp_path / 'empty-catalog.sqlite3'),
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window.Show()
        wx.Yield()
        panel = window._ScrubberFrame__image_panel
        panel.add_new_box((5, 5, 30, 30))
        panel.add_new_box((45, 5, 30, 30))
        for coords in ((5, 45, 30, 30), (45, 45, 30, 30)):
            box = BoxData(coords, [], 'automatic')
            window._ScrubberFrame__frame_boxes[0].append(box)
            window.mark_not_pin(box)
        wx.Yield()

        examples = window._feedback_store.examples()
        positives = [item for item in examples if item.label == 'positive']
        assert len(positives) == 2
        assert all(item.pin_id is None and item.tags == () for item in positives)

        tag_panel = window._ScrubberFrame__tag_panel
        window.GetEventHandler().ProcessEvent(wx.CommandEvent(
            wx.EVT_MENU.typeId, window._training_add_item.GetId()))
        status = tag_panel.training_status.GetLabel()
        assert '4 examples indexed' in status
        assert '2 pin crops (linked frames: 0, 2 without verified ID)' in status
        assert '0 added now' in status

        window.GetEventHandler().ProcessEvent(wx.CommandEvent(
            wx.EVT_MENU.typeId, window._training_run_item.GetId()))
        deadline = monotonic() + 10
        while window._model_training_future is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        wx.Yield()
        assert window._model_training_future is None
        assert window._feedback_model.trained_model is not None
        assert window._feedback_model.trained_model.sample_counts == {
            'positive': 2, 'negative': 2}
    finally:
        window.pause_playback()
        window.Destroy()
        wx.Yield()
