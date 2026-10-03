from io import StringIO
from unittest.mock import patch
import json

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from controls.imagepanel import ImagePanel
from detection_feedback import FeedbackModel, FeedbackStore, make_feedback_example
from pin_detection import detect_pin_boxes
from scrubberframe import (load_boxes_from_stream, save_boxes_to_file,
                           save_boxes_to_stream)
from videoscrubber import VideoScrubber


def sample_frame() -> np.ndarray:
    image = np.full((100, 190, 3), 135, dtype=np.uint8)
    image[15:55, 10:50] = np.random.default_rng(41).integers(10, 245, (40, 40, 3), dtype=np.uint8)
    image[15:55, 110:150] = np.random.default_rng(42).integers(10, 245, (40, 40, 3), dtype=np.uint8)
    return image


def test_confirmations_and_rejections_train_and_persist_candidate_ranker(tmp_path):
    image = sample_frame()
    positive = (10, 15, 40, 40)
    negative = (110, 15, 40, 40)
    path = tmp_path / 'feedback.sqlite3'
    store = FeedbackStore(path)
    store.save(make_feedback_example('positive-id', 'positive', image, positive,
                                     'video.mp4', 2, 'track-a', 'pinpanion:7'))
    assert FeedbackModel(store).filter_and_rank(image, [negative, positive]) == [
        positive, negative]
    store.save(make_feedback_example('negative-id', 'negative', image, negative,
                                     'video.mp4', 2, 'track-b', None))
    model = FeedbackModel(store)

    assert model.filter_and_rank(image, [negative, positive]) == [positive]
    assert [(item.label, item.frame_index, item.coords) for item in store.examples()] == [
        ('positive', 2, positive), ('negative', 2, negative)]
    changed = image.copy()
    changed[15:55, 110:150] = np.random.default_rng(43).integers(
        10, 245, (40, 40, 3), dtype=np.uint8)
    near_negative = image.copy()
    near_negative[15:55, 110:150] = np.uint8(
        0.7 * image[15:55, 110:150].astype(np.float32) +
        0.3 * changed[15:55, 110:150].astype(np.float32))
    assert negative not in model.filter_and_rank(near_negative, [negative, positive])
    assert negative in model.filter_and_rank(changed, [negative, positive])
    assert store.db.execute('SELECT DISTINCT rights_status FROM feedback_examples').fetchall() == [
        ('unverified',)]
    store.close()

    reopened = FeedbackStore(path)
    assert FeedbackModel(reopened).filter_and_rank(image, [negative, positive]) == [positive]
    reopened.close()


def test_low_detail_feedback_is_saved_but_reported_as_unusable():
    image = np.full((80, 80, 3), 130, dtype=np.uint8)
    store = FeedbackStore()
    try:
        store.save(make_feedback_example('flat', 'positive', image, (10, 10, 40, 40),
                                         'video.mp4', 0, 'track', None))
        model = FeedbackModel(store)
        assert store.counts() == {'positive': 1, 'negative': 0}
        assert model.sample_counts == {'positive': 0, 'negative': 0}
    finally:
        store.close()


@pytest.mark.gui
def test_context_menu_separates_area_removal_from_negative_review(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = sample_frame()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    window = VideoScrubber(None, 'Removal choices', '', image_array=[bgr, bgr],
                           feedback_path=tmp_path / 'feedback.sqlite3',
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        selected = BoxData((10, 15, 40, 40), ['Possible pin'], 'automatic',
                           track_id='same', pin_id='pinpanion:99',
                           match_confidence=0.62)
        linked = BoxData(selected.coords, ['Possible pin'], 'automatic',
                         track_id='same')
        window._ScrubberFrame__frame_boxes.update({0: [selected], 1: [linked]})
        window.display_image(reconcile=False)
        panel: ImagePanel = window._ScrubberFrame__image_panel
        menu = panel.make_box_context_menu(selected)
        assert [item.GetItemLabelText() for item in menu.GetMenuItems()] == [
            'Confirm is a pin', 'Confirm pin identity', 'Not this pin',
            'Remove area', 'Not a pin']
        assert [item.GetItemLabel() for item in menu.GetMenuItems()][2:] == [
            'Not this pin (&X)', 'Remove area\tDel', '&Not a pin']
        remove = menu.GetMenuItems()[3]
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, remove.GetId()))
        menu.Destroy()
        assert window._ScrubberFrame__frame_boxes[0] == []
        assert linked.review_state == 'unconfirmed'
        assert window._feedback_store.examples() == []

        window.undo_action()
        assert window._ScrubberFrame__frame_boxes[0] == [selected]
        menu = panel.make_box_context_menu(selected)
        not_pin = menu.GetMenuItems()[4]
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, not_pin.GetId()))
        menu.Destroy()
        assert selected.review_state == linked.review_state == 'rejected'
        assert selected.pin_id == 'pinpanion:99'
        assert selected.match_confidence == 0.62
        assert panel.boxes == [selected]
        assert window._ScrubberFrame__tag_panel.boxes == []
        examples = window._feedback_store.examples()
        assert len(examples) == 1
        assert examples[0].label == 'negative' and examples[0].pin_id is None
        window.undo_action()
        assert selected.review_state == linked.review_state == 'unconfirmed'
        assert window._feedback_store.examples() == []

        manual = BoxData((60, 15, 30, 30), ['Manual'], 'user')
        menu = panel.make_box_context_menu(manual)
        assert [item.GetItemLabelText() for item in menu.GetMenuItems()] == [
            'Confirm is a pin', 'Confirm pin identity', 'Not this pin', 'Remove area']
        menu.Destroy()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_not_a_pin_replaces_prior_positive_review_on_automatic_track(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = sample_frame()
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    window = VideoScrubber(None, 'Correct false positive', '', image_array=[bgr, bgr],
                           feedback_path=tmp_path / 'feedback.sqlite3',
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        direct = BoxData((10, 15, 40, 40), ['Wrong pin'], 'automatic',
                         track_id='same', pin_id='pinpanion:99')
        linked = BoxData(direct.coords, ['Wrong pin'], 'automatic',
                         track_id='same', review_state='inherited')
        window._ScrubberFrame__frame_boxes.update({0: [direct], 1: [linked]})
        window.display_image(reconcile=False)
        window.confirm_box(direct)
        assert [item.label for item in window._feedback_store.examples()] == ['positive']
        window.mark_not_pin(direct)
        assert direct.review_state == linked.review_state == 'rejected'
        examples = window._feedback_store.examples()
        assert len(examples) == 1 and examples[0].label == 'negative'
        assert examples[0].pin_id is None
        window.undo_action()
        assert direct.review_state == 'confirmed'
        assert linked.review_state == 'inherited'
        assert [item.label for item in window._feedback_store.examples()] == ['positive']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_gui_feedback_changes_future_detection_and_undo_redo(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = sample_frame()
    frames = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR)] * 2
    path = tmp_path / 'feedback.sqlite3'
    window = VideoScrubber(None, 'Feedback test', '', image_array=frames,
                           feedback_path=path, load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        confirmed = BoxData((10, 15, 40, 40), [], 'automatic')
        rejected = BoxData((110, 15, 40, 40), ['Wrong suggestion'], 'automatic',
                           pin_id='pinpanion:99', match_confidence=0.71)
        window._ScrubberFrame__frame_boxes[0] = [confirmed, rejected]
        window.display_image()
        initial_revision = window._feedback_model.version
        window.confirm_box(confirmed)
        assert window._feedback_model.version > initial_revision
        assert window._feedback_model.sample_counts == {'positive': 1, 'negative': 0}
        assert 'pin 1/1, non-pin 0/0 usable' in window.GetStatusBar().GetStatusText(0)
        window.mark_not_pin(rejected)
        assert window._ScrubberFrame__frame_boxes[0][1] is rejected
        assert rejected.review_state == 'rejected'
        assert rejected.tags == ['Wrong suggestion']
        assert rejected.pin_id == 'pinpanion:99'
        assert rejected.match_confidence == 0.71
        saved = StringIO()
        save_boxes_to_stream(saved, window._ScrubberFrame__frame_boxes)
        restored = load_boxes_from_stream(StringIO(saved.getvalue()))[0][1]
        assert restored.review_state == 'rejected'
        assert restored.coords == rejected.coords
        assert restored.tags == rejected.tags and restored.pin_id == rejected.pin_id
        assert restored.feedback_label == 'negative'
        negative_example = window._feedback_store.get(rejected.feedback_id)
        assert negative_example is not None and negative_example.label == 'negative'
        assert negative_example.coords == rejected.coords
        assert negative_example.pin_id is None
        assert np.array_equal(cv2.cvtColor(cv2.imdecode(
            np.frombuffer(negative_example.crop_png, dtype=np.uint8), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB), image[15:55, 110:150])
        assert [item.label for item in window._feedback_store.examples()] == [
            'positive', 'negative']
        assert window._feedback_model.sample_counts == {'positive': 1, 'negative': 1}
        assert 'pin 1/1, non-pin 1/1 usable' in window.GetStatusBar().GetStatusText(0)
        info = window._ScrubberFrame__tag_panel.feedback_info_btn
        with patch('scrubberframe.wx.MessageBox') as show_info:
            info.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId, info.GetId()))
        explanation = show_info.call_args.args[0]
        assert 'Confirmed pins: 1 saved, 1 usable' in explanation
        assert 'Rejected automatic candidates: 1 saved, 1 usable' in explanation
        assert 'not a trained neural detector' in explanation

        with patch('pin_detection.propose_regions', return_value=[rejected.coords, confirmed.coords]), \
             patch('pin_detection.refine_region', return_value=None):
            assert detect_pin_boxes(image, feedback_model=window._feedback_model) == [confirmed.coords]

        with patch('pin_detection.propose_regions', return_value=[rejected.coords, confirmed.coords]), \
             patch('pin_detection.refine_region', return_value=None), \
             patch('scrubberframe.detect_pin_boxes', wraps=detect_pin_boxes) as detect:
            window.on_next(None)
        assert detect.call_args.kwargs['feedback_model'] is window._feedback_model
        assert len(window._feedback_store.examples()) == 2
        window.goto_frame(0)

        window.undo_action()
        assert [item.label for item in window._feedback_store.examples()] == ['positive']
        assert window._feedback_model.sample_counts == {'positive': 1, 'negative': 0}
        window.redo_action()
        assert [item.label for item in window._feedback_store.examples()] == [
            'positive', 'negative']
        assert window._feedback_model.sample_counts == {'positive': 1, 'negative': 1}

        window.remove_box(confirmed)
        assert [item.label for item in window._feedback_store.examples()] == ['negative']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_loading_legacy_reviewed_boxes_seeds_feedback_once_per_rejected_track(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = sample_frame()
    path = tmp_path / 'feedback.sqlite3'
    annotations = tmp_path / 'legacy.json'
    annotations.write_text(json.dumps({
        '0': [
            {'coords': [10, 15, 40, 40], 'tags': ['Mario'],
             'review_state': 'confirmed', 'track_id': 'pin'},
            {'coords': [110, 15, 40, 40], 'tags': [],
             'review_state': 'rejected', 'track_id': 'false'},
        ],
        '1': [
            {'coords': [110, 15, 40, 40], 'tags': [],
             'review_state': 'rejected', 'track_id': 'false'},
        ],
    }), encoding='utf-8')
    window = VideoScrubber(None, 'Legacy feedback test', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)] * 2,
                           feedback_path=path, load_catalog=False)
    try:
        window._feedback_source = 'legacy-video.mp4'
        window.box_data_filename = str(annotations)
        window.load_box_data()
        assert [item.label for item in window._feedback_store.examples()] == [
            'positive', 'negative']
        assert window._ScrubberFrame__frame_boxes[1][0].feedback_label is None
        window.load_box_data()
        assert len(window._feedback_store.examples()) == 2
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_rejected_annotation_reloads_its_negative_crop_and_original_guess(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = sample_frame()
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    annotations = tmp_path / 'rejected.json'
    first = VideoScrubber(None, 'Original rejection', '', image_array=[frame],
                          feedback_path=tmp_path / 'first.sqlite3', load_catalog=False)
    try:
        first.Show()
        wx.Yield()
        box = BoxData((110, 15, 40, 40), ['Wrong suggestion'], 'automatic',
                      pin_id='pinpanion:99', match_confidence=0.71)
        first._ScrubberFrame__frame_boxes[0] = [box]
        first.mark_not_pin(box)
        save_boxes_to_file(str(annotations), first._ScrubberFrame__frame_boxes)
    finally:
        first.Destroy()
        wx.Yield()

    restored = VideoScrubber(None, 'Reloaded rejection', '', image_array=[frame],
                             feedback_path=tmp_path / 'second.sqlite3', load_catalog=False)
    try:
        restored.box_data_filename = str(annotations)
        restored.load_box_data()
        loaded = restored._ScrubberFrame__frame_boxes[0][0]
        assert loaded.review_state == 'rejected'
        assert loaded.coords == (110, 15, 40, 40)
        assert loaded.tags == ['Wrong suggestion']
        assert loaded.pin_id == 'pinpanion:99' and loaded.match_confidence == 0.71
        examples = restored._feedback_store.examples()
        assert len(examples) == 1
        assert examples[0].label == 'negative' and examples[0].pin_id is None
        assert restored._feedback_model.sample_counts == {'positive': 0, 'negative': 1}
    finally:
        restored.Destroy()
        wx.Yield()
