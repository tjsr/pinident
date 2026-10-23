from pathlib import Path
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pytest
import wx

from boxdata import BoxData
from controls.classifierreviewdialog import ClassifierReviewDialog
from detection_feedback import FeedbackExample
from frame_presence import FramePresence
from pin_catalog import PinMatch
from scrubberframe import (load_boxes_from_file, save_boxes_to_file,
                           save_review_feedback)
from videoscrubber import VideoScrubber


@pytest.mark.gui
def test_review_mode_defers_boxes_rejected_after_queue_opened(tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    frames = [np.full((80, 80, 3), value, dtype=np.uint8)
              for value in (30, 60, 90)]
    window = VideoScrubber(None, 'Deferred negatives', '', image_array=frames,
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        first = BoxData((5, 5, 30, 30), [], 'automatic')
        newly_rejected = BoxData((5, 5, 30, 30), [], 'automatic')
        remaining = BoxData((5, 5, 30, 30), [], 'automatic')
        window._ScrubberFrame__frame_boxes.update({
            0: [first], 1: [newly_rejected], 2: [remaining]})
        dialog = ClassifierReviewDialog(window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()

        newly_rejected.review_state = 'rejected'
        dialog._on_next(wx.CommandEvent())
        assert dialog._current_area is not None
        assert dialog._current_area.box is remaining
        dialog._on_next(wx.CommandEvent())
        assert dialog._current_area is not None
        assert dialog._current_area.box is newly_rejected
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


def wait_for_save(window: VideoScrubber, filename: Path) -> None:
    if window._review_save_call is not None:
        window._review_save_call.Stop()
    window._flush_review_save()
    deadline = monotonic() + 5
    while monotonic() < deadline:
        wx.Yield()
        if window._review_feedback_changes and window._review_feedback_future is None:
            if window._review_feedback_call is not None:
                window._review_feedback_call.Stop()
            window._flush_review_feedback()
        if window._review_save_pending and window._review_save_future is None:
            if window._review_save_call is not None:
                window._review_save_call.Stop()
            window._flush_review_save()
        if (filename.exists() and window._review_save_future is None and
                not window._review_save_pending and
                window._review_feedback_future is None and
                not window._review_feedback_changes):
            return
        sleep(0.005)
    pytest.fail(f'Background review save did not complete: '
                f'exists={filename.exists()} pending={window._review_save_pending} '
                f'future={window._review_save_future} call={window._review_save_call}')


@pytest.mark.gui
def test_review_mode_advances_before_background_save(tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    frames = [np.full((90, 90, 3), value, dtype=np.uint8)
              for value in (40, 100)]
    window = VideoScrubber(None, 'Review saved crops', '', image_array=frames,
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        candidate = BoxData((10, 10, 40, 40), [], 'automatic')
        later = BoxData((20, 20, 30, 30), [], 'automatic')
        window._ScrubberFrame__frame_boxes.update({0: [candidate], 1: [later]})
        dialog = ClassifierReviewDialog(
            window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()
        pin_tip = dialog._action_buttons['confirm_pin'].GetToolTipText()
        restore_tip = dialog._action_buttons['restore_pin'].GetToolTipText()
        assert 'without confirming its suggested name or catalog ID' in pin_tip
        assert 'positive pin crop for training' in pin_tip
        assert 'replaces its negative review with a positive pin crop' in restore_tip
        assert 'clears the old pin identity guess' in restore_tip
        assert (dialog._status.GetPosition().y + dialog._status.GetSize().height <=
                dialog.GetClientSize().height)
        assert dialog._current_area is not None
        assert dialog._current_area.box is candidate

        dialog._act('confirm_pin')
        assert dialog._position == 1

        assert candidate.review_state == 'confirmed'
        assert candidate.feedback_label == 'positive'
        assert dialog._current_area is not None
        assert dialog._current_area.box is later
        assert window.current_index == 1
        assert not (tmp_path / 'video.json').exists()
        assert window._feedback_store.counts()['positive'] == 0
        wait_for_save(window, tmp_path / 'video.json')
        saved = load_boxes_from_file(str(tmp_path / 'video.json'))
        assert saved[0][0].review_state == 'confirmed'
        assert saved[0][0].feedback_label == 'positive'
        assert window._feedback_store.counts()['positive'] == 1
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_review_mode_rejects_suggested_identity_but_keeps_positive_pin(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    frame = np.full((90, 90, 3), 80, dtype=np.uint8)
    window = VideoScrubber(None, 'Reject identity in review', '',
                           image_array=[frame], load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        candidate = BoxData((10, 10, 50, 50), ['Wrong pin'], 'automatic',
                            pin_id='pin-wrong', match_confidence=0.45)
        window._ScrubberFrame__frame_boxes[0] = [candidate]
        dialog = ClassifierReviewDialog(
            window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()

        reject_key = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        reject_key.SetKeyCode(ord('N'))
        with patch('controls.classifierreviewdialog.wx.Window.FindFocus',
                   return_value=None):
            dialog._on_key(reject_key)
        assert dialog._position == 1

        assert candidate.review_state == 'confirmed'
        assert candidate.pin_id is None and candidate.tags == []
        assert candidate.feedback_label == 'positive'
        assert [review.pin_id for review in candidate.identity_rejections] == [
            'pin-wrong']
        wait_for_save(window, tmp_path / 'video.json')
        saved = load_boxes_from_file(str(tmp_path / 'video.json'))[0][0]
        assert saved.review_state == 'confirmed' and saved.pin_id is None
        assert saved.feedback_label == 'positive'
        assert [review.pin_id for review in saved.identity_rejections] == [
            'pin-wrong']
        assert window._feedback_store.counts()['positive'] == 1
        assert window._feedback_store.identity_rejection_count(
            window._feedback_source) == 1
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_review_mode_y_verifies_linked_identity_in_current_frame(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    frame = np.full((90, 90, 3), 90, dtype=np.uint8)
    window = VideoScrubber(None, 'Verify linked crop', '',
                           image_array=[frame], load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        linked = BoxData((10, 10, 50, 50), ['Mario'], 'automatic',
                         pin_id='pin-7', review_state='inherited',
                         identity_confirmed=True)
        window._ScrubberFrame__frame_boxes[0] = [linked]
        dialog = ClassifierReviewDialog(window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()
        confirm_key = wx.KeyEvent(wx.wxEVT_CHAR_HOOK)
        confirm_key.SetKeyCode(ord('Y'))
        with patch('controls.classifierreviewdialog.wx.Window.FindFocus',
                   return_value=None):
            dialog._on_key(confirm_key)
        assert dialog._position == 1

        assert linked.review_state == 'confirmed' and linked.identity_confirmed
        assert linked.feedback_label == 'positive'
        wait_for_save(window, tmp_path / 'video.json')
        saved = load_boxes_from_file(str(tmp_path / 'video.json'))[0][0]
        assert saved.pin_id == 'pin-7' and saved.identity_confirmed
        assert saved.feedback_label == 'positive'
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_review_mode_handles_negative_restore_and_catalog_choice(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    frame = np.full((120, 120, 3), 100, dtype=np.uint8)
    window = VideoScrubber(None, 'Review decisions', '',
                           image_array=[frame], load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        negative = BoxData((0, 0, 30, 30), [], 'automatic')
        restore = BoxData((40, 0, 30, 30), ['Bad guess'], 'automatic',
                          review_state='rejected', feedback_id='old-negative',
                          feedback_label='negative', pin_id='pin-bad')
        identified = BoxData((80, 0, 30, 30), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [negative, restore, identified]
        dialog = ClassifierReviewDialog(
            window, window.classifier_review_areas(), [('pin-7', 'Mario')])
        dialog.Show()
        wx.Yield()

        dialog._act('not_pin')
        assert dialog._position == 1
        assert negative.review_state == 'rejected'
        assert negative.feedback_label == 'negative'
        assert dialog._current_area is not None
        assert dialog._current_area.box is identified

        dialog._search.SetValue('mario')
        assert len(dialog._choices) == 1
        dialog._results.SetSelection(0)
        dialog._on_choose(wx.CommandEvent())
        assert dialog._position == 2
        assert identified.pin_id == 'pin-7'
        assert identified.tags == ['Mario']
        assert identified.review_state == 'confirmed'
        assert identified.identity_confirmed
        assert dialog._current_area is not None
        assert dialog._current_area.box is restore

        dialog._act('restore_pin')
        assert dialog._position == 3
        assert restore.review_state == 'confirmed'
        assert restore.pin_id is None and restore.tags == []
        assert restore.feedback_label == 'positive'
        wait_for_save(window, tmp_path / 'video.json')
        saved = load_boxes_from_file(str(tmp_path / 'video.json'))[0]
        assert [box.review_state for box in saved] == [
            'rejected', 'confirmed', 'confirmed']
        assert [box.feedback_label for box in saved] == [
            'negative', 'positive', 'positive']
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('selection_mode', [
    'ranked', 'search', 'exact-search', 'partial-search', 'ambiguous-search'])
@pytest.mark.parametrize('confirmation', ['identity-button', 'selection-button'])
def test_confirm_identity_uses_explicit_catalog_selection(
        tmp_path: Path, selection_mode: str, confirmation: str) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Correct suggested identity', '',
        image_array=[np.full((90, 90, 3), 90, dtype=np.uint8)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 50, 50), ['Wrong pin'], 'automatic',
                      pin_id='pin-wrong', match_confidence=0.8)
        window._ScrubberFrame__frame_boxes[0] = [box]
        dialog = ClassifierReviewDialog(window, window.classifier_review_areas(), [
            ('pin-wrong', 'Wrong pin'), ('pin-right', 'Right pin')])
        dialog.Show()
        wx.Yield()
        if selection_mode != 'ranked':
            query = ('Right' if selection_mode == 'partial-search' else
                     'pin' if selection_mode == 'ambiguous-search' else 'Right pin')
            dialog._search.SetValue(query)
            assert [choice.pin_id for choice in dialog._choices] == (
                ['pin-wrong', 'pin-right'] if selection_mode == 'ambiguous-search'
                else ['pin-right'])
            if selection_mode == 'search':
                dialog._results.SetSelection(0)
            else:
                assert dialog._results.GetSelection() == wx.NOT_FOUND
        else:
            dialog._choices = [
                PinMatch('pin-wrong', 'Wrong pin', 0),
                PinMatch('pin-right', 'Right pin', 0)]
            dialog._results.SetItems(['Wrong pin', 'Right pin'])
            dialog._results.SetSelection(1)

        if selection_mode == 'ambiguous-search':
            button = dialog._action_buttons[
                'choose_identity' if confirmation == 'selection-button'
                else 'confirm_identity']
            button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId()))
            assert box.pin_id == 'pin-wrong'
            assert dialog._position == 0
            dialog._results.SetSelection(1)
        button = dialog._action_buttons[
            'choose_identity' if confirmation == 'selection-button'
            else 'confirm_identity']
        button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId()))
        assert dialog._position == 1
        assert box.pin_id == 'pin-right'
        assert box.tags == ['Right pin']
        assert box.identity_confirmed
        assert [review.pin_id for review in box.identity_rejections] == ['pin-wrong']
        wait_for_save(window, tmp_path / 'video.json')
        saved = load_boxes_from_file(str(tmp_path / 'video.json'))[0][0]
        assert saved.pin_id == 'pin-right'
        assert saved.tags == ['Right pin']
        assert window._feedback_store.identity_rejection_count(
            window._feedback_source) == 1
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_review_mode_background_save_failure_keeps_review_and_allows_retry(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Review save failure', '',
        image_array=[np.full((80, 80, 3), 80, dtype=np.uint8)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        dialog = ClassifierReviewDialog(window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()
        dialog._act('confirm_pin')
        assert dialog._position == 1
        with patch('scrubberframe.save_boxes_to_file', side_effect=OSError('disk full')):
            if window._review_save_call is not None:
                window._review_save_call.Stop()
            window._flush_review_save()
            future = window._review_save_future
            assert future is not None
            with pytest.raises(OSError, match='disk full'):
                future.result(timeout=5)
            wx.Yield()
        assert window._review_save_pending
        assert 'Background save failed' in window.GetStatusBar().GetStatusText(1)
        assert window.save_annotations()
        assert dialog._position == 1
        assert load_boxes_from_file(str(tmp_path / 'video.json'))[0][0].review_state == 'confirmed'
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_later_review_is_saved_after_an_older_background_write(tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Review save ordering', '',
        image_array=[np.full((80, 80, 3), 80, dtype=np.uint8)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    filename = tmp_path / 'video.json'
    window.box_data_filename = str(filename)
    dialog: ClassifierReviewDialog | None = None
    started = Event()
    release = Event()
    writes = 0

    def delayed_save(path: str, boxes: dict[int, list[BoxData]],
                     presence: dict[int, FramePresence] | None = None) -> None:
        nonlocal writes
        writes += 1
        if writes == 1:
            started.set()
            assert release.wait(timeout=5)
        save_boxes_to_file(path, boxes, presence)

    try:
        window.Show()
        wx.Yield()
        first = BoxData((0, 0, 30, 30), [], 'automatic')
        second = BoxData((40, 0, 30, 30), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [first, second]
        dialog = ClassifierReviewDialog(window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()
        with patch('scrubberframe.save_boxes_to_file', side_effect=delayed_save):
            dialog._act('confirm_pin')
            if window._review_save_call is not None:
                window._review_save_call.Stop()
            window._flush_review_save()
            assert started.wait(timeout=5)
            dialog._act('not_pin')
            assert dialog._position == 2
            assert second.review_state == 'rejected'
            release.set()
            deadline = monotonic() + 5
            while window._review_save_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            wait_for_save(window, filename)
        saved = load_boxes_from_file(str(filename))[0]
        assert [box.review_state for box in saved] == ['confirmed', 'rejected']
        assert writes == 2
    finally:
        release.set()
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_background_feedback_failure_retries_on_explicit_save(tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Review feedback save failure', '',
        image_array=[np.full((80, 80, 3), 80, dtype=np.uint8)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    dialog: ClassifierReviewDialog | None = None
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        dialog = ClassifierReviewDialog(window, window.classifier_review_areas(), [])
        dialog.Show()
        wx.Yield()
        dialog._act('confirm_pin')
        assert dialog._position == 1
        assert window._feedback_store.counts()['positive'] == 0
        with patch('scrubberframe.save_review_feedback',
                   side_effect=OSError('database unavailable')):
            if window._review_feedback_call is not None:
                window._review_feedback_call.Stop()
            window._flush_review_feedback()
            future = window._review_feedback_future
            assert future is not None
            with pytest.raises(OSError, match='database unavailable'):
                future.result(timeout=5)
            wx.Yield()
        assert window._review_feedback_failed
        assert window._review_feedback_changes
        assert window.save_annotations()
        assert window._feedback_store.counts()['positive'] == 1
        assert load_boxes_from_file(str(tmp_path / 'video.json'))[0][0].review_state == 'confirmed'
    finally:
        if dialog is not None:
            dialog.Destroy()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_newer_feedback_review_wins_while_older_batch_is_writing(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Review feedback ordering', '',
        image_array=[np.full((80, 80, 3), 80, dtype=np.uint8)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    started = Event()
    release = Event()
    writes = 0

    def delayed_feedback(
            path: str, source: str,
            changes: dict[str, FeedbackExample | None]) -> None:
        nonlocal writes
        writes += 1
        if writes == 1:
            started.set()
            assert release.wait(timeout=5)
        save_review_feedback(path, source, changes)

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False, catalog_matching=False)
        with patch('scrubberframe.save_review_feedback', side_effect=delayed_feedback):
            with window.classifier_review_action():
                window.confirm_box(box)
            if window._review_feedback_call is not None:
                window._review_feedback_call.Stop()
            window._flush_review_feedback()
            assert started.wait(timeout=5)
            with window.classifier_review_action():
                window.mark_not_pin(box)
            assert box.feedback_label == 'negative'
            release.set()
            deadline = monotonic() + 5
            while window._review_feedback_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            assert window.save_annotations()
        assert writes == 2
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 1}
        assert load_boxes_from_file(str(tmp_path / 'video.json'))[0][0].review_state == 'rejected'
    finally:
        release.set()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_undo_after_review_cannot_be_overwritten_by_pending_feedback(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Undo queued feedback', '',
        image_array=[np.full((80, 80, 3), 80, dtype=np.uint8)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    window.box_data_filename = str(tmp_path / 'video.json')
    started = Event()
    release = Event()

    def delayed_feedback(
            path: str, source: str,
            changes: dict[str, FeedbackExample | None]) -> None:
        started.set()
        assert release.wait(timeout=5)
        save_review_feedback(path, source, changes)

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False, catalog_matching=False)
        with patch('scrubberframe.save_review_feedback', side_effect=delayed_feedback):
            with window.classifier_review_action():
                window.confirm_box(box)
            window.queue_review_save()
            if window._review_feedback_call is not None:
                window._review_feedback_call.Stop()
            window._flush_review_feedback()
            assert started.wait(timeout=5)
            window.undo_action()
            assert box.review_state == 'unconfirmed'
            release.set()
            assert window.save_annotations()
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 0}
        assert load_boxes_from_file(str(tmp_path / 'video.json'))[0][0].review_state == 'unconfirmed'
    finally:
        release.set()
        window.Destroy()
        wx.Yield()
