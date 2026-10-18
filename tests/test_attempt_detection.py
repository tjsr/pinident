from threading import Event
from time import monotonic, sleep

import numpy as np
import pytest
import wx
import cv2

from boxdata import BoxData
from controls.imagepanel import ImagePanel
from detection_feedback import FeedbackModel, FeedbackStore, make_feedback_example
from pin_catalog import PinMatch
from pin_detection import identify_candidates
from videoscrubber import VideoScrubber


def wait_for_attempt(window):
    deadline = monotonic() + 5
    while window._pending_attempt is not None and monotonic() < deadline:
        wx.Yield()
        sleep(0.005)
    assert window._pending_attempt is None


def activate_attempt(menu):
    item = next(item for item in menu.GetMenuItems()
                if item.GetItemLabelText() == 'Attempt detection')
    menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item.GetId()))


def test_explicit_retry_keeps_geometric_catalog_match_over_reviewed_crop():
    rng = np.random.default_rng(81)
    image = rng.integers(15, 235, (100, 100, 3), dtype=np.uint8)
    store = FeedbackStore()
    try:
        store.save(make_feedback_example('reviewed', 'positive', image,
                                         (0, 0, 100, 100), 'source.mp4', 0,
                                         'track', 'pin-reviewed', ('Reviewed',)))
        feedback = FeedbackModel(store)

        class GeometricMatcher:
            def suggest(self, _crop):
                return PinMatch('pin-geometric', 'Geometric', 12, 0.75)

            match = suggest

        box = BoxData((0, 0, 100, 100), [], 'automatic',
                      review_state='confirmed')
        identify_candidates(image, [box], GeometricMatcher(), best_effort=True,
                            feedback_model=feedback,
                            prefer_reviewed_identity=True)
        assert (box.pin_id, box.tags) == ('pin-geometric', ['Geometric'])
    finally:
        store.close()


@pytest.mark.gui
def test_multi_selection_menu_targets_the_clicked_rectangle():
    app = wx.App.Get() or wx.App(False)
    parent = wx.Frame(None)
    panel = ImagePanel(parent)
    first = BoxData((0, 0, 20, 20), [], 'automatic')
    clicked = BoxData((25, 0, 20, 20), [], 'automatic')
    requested = []
    panel.on_attempt_detection = requested.append
    try:
        menu = panel.make_selection_context_menu([first, clicked], clicked)
        activate_attempt(menu)
        menu.Destroy()
        assert requested == [clicked]
    finally:
        parent.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_updates_only_clicked_box_and_is_undoable(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Attempt one area', '', image_array=[image],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')

    class Matcher:
        has_references = True

        def suggest(self, crop):
            assert crop.shape == (30, 30, 3)
            return PinMatch('pin-new', 'New Pin', 7, 0.72)

        match = suggest

    try:
        window.Show()
        wx.Yield()
        selected = BoxData((10, 15, 30, 30), ['Old guess'], 'automatic',
                           pin_id='pin-old', match_confidence=0.20)
        other = BoxData((60, 15, 20, 20), ['Other'], 'automatic',
                        pin_id='pin-other', match_confidence=0.65)
        window._ScrubberFrame__frame_boxes[0] = [selected, other]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = Matcher()
        window.display_image(reconcile=False)

        menu = window._ScrubberFrame__image_panel.make_box_context_menu(selected)
        activate_attempt(menu)
        menu.Destroy()
        wait_for_attempt(window)

        assert window._ScrubberFrame__frame_boxes[0][0] is selected
        assert (selected.coords, selected.pin_id, selected.tags,
                selected.match_confidence, selected.identity_confirmed) == (
                    (10, 15, 30, 30), 'pin-new', ['New Pin'], 0.72, False)
        assert (other.pin_id, other.tags) == ('pin-other', ['Other'])
        assert window._feedback_store.examples() == []
        window.undo_action()
        assert (selected.pin_id, selected.tags, selected.match_confidence) == (
            'pin-old', ['Old guess'], 0.20)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_uses_reviewed_identity_for_unconfirmed_area(tmp_path):
    app = wx.App.Get() or wx.App(False)
    rng = np.random.default_rng(12)
    image = rng.integers(15, 235, (180, 180, 3), dtype=np.uint8)
    cv2.circle(image, (90, 90), 72, (220, 35, 30), 5)
    feedback_path = tmp_path / 'feedback.sqlite3'
    store = FeedbackStore(feedback_path)
    store.save(make_feedback_example('known', 'positive', image,
                                     (0, 0, 180, 180), 'source.mp4', 0,
                                     'reviewed-track', 'pin-known', ('Known Pin',)))
    store.close()
    window = VideoScrubber(None, 'Detect reviewed identity', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False, feedback_path=feedback_path)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((0, 0, 180, 180), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        activate_attempt(menu)
        menu.Destroy()
        wait_for_attempt(window)
        assert (box.pin_id, box.tags) == ('pin-known', ['Known Pin'])
        assert box.review_state == 'unconfirmed' and not box.identity_confirmed
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_prefers_reviewed_identity_over_colour_only_catalog_guess(tmp_path):
    app = wx.App.Get() or wx.App(False)
    rng = np.random.default_rng(44)
    image = rng.integers(15, 235, (180, 180, 3), dtype=np.uint8)
    cv2.circle(image, (90, 90), 70, (210, 35, 30), 5)
    feedback_path = tmp_path / 'feedback.sqlite3'
    store = FeedbackStore(feedback_path)
    store.save(make_feedback_example('reviewed', 'positive', image,
                                     (0, 0, 180, 180), 'source.mp4', 0,
                                     'known-track', 'pin-known', ('Known Pin',)))
    store.close()
    window = VideoScrubber(None, 'Retry with reviewed ID', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False, feedback_path=feedback_path)

    class ColourOnlyMatcher:
        has_references = True

        def suggest(self, _crop):
            return PinMatch('pin-wrong', 'Wrong Pin', 0, 0.85)

        def match(self, _crop):
            return None

    try:
        window.Show()
        wx.Yield()
        box = BoxData((0, 0, 180, 180), ['Wrong Pin'], 'automatic',
                      pin_id='pin-wrong', match_confidence=0.85)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = ColourOnlyMatcher()
        window.display_image(reconcile=False)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        activate_attempt(menu)
        menu.Destroy()
        wait_for_attempt(window)
        assert (box.pin_id, box.tags) == ('pin-known', ['Known Pin'])
        assert not box.identity_confirmed
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_explains_when_current_guess_is_still_best(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Same best guess', '',
                           image_array=[np.zeros((80, 80, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')

    class Matcher:
        has_references = True

        def suggest(self, _crop):
            return PinMatch('pin-same', 'Same Pin', 0, 0.74)

        match = suggest

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), ['Same Pin'], 'automatic',
                      pin_id='pin-same', match_confidence=0.74)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = Matcher()
        window.display_image(reconcile=False)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        activate_attempt(menu)
        menu.Destroy()
        wait_for_attempt(window)
        assert 'already the best match' in window.GetStatusBar().GetStatusText(1)
        assert (box.pin_id, box.tags) == ('pin-same', ['Same Pin'])
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_reports_when_no_identity_sources_are_loaded(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'No identity sources', '',
                           image_array=[np.zeros((80, 80, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        activate_attempt(menu)
        menu.Destroy()
        assert window._pending_attempt is None
        assert 'no catalog images or reviewed pin identities' in (
            window.GetStatusBar().GetStatusText(1))
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('box', [
    BoxData((10, 15, 30, 30), ['My label'], 'user', review_state='confirmed',
            identity_confirmed=False),
    BoxData((10, 15, 30, 30), ['Typed label'], 'automatic',
            identity_confirmed=False),
    BoxData((10, 15, 30, 30), ['Verified'], 'automatic',
            review_state='confirmed', pin_id='pin-verified', identity_confirmed=True),
])
def test_attempt_detection_preserves_authoritative_identity(tmp_path, box):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'Preserve review', '',
                           image_array=[np.zeros((90, 120, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')

    class Matcher:
        has_references = True

        def suggest(self, _crop):
            return PinMatch('pin-new', 'New Pin', 7, 0.72)

        match = suggest

    try:
        window.Show()
        wx.Yield()
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = Matcher()
        window.display_image(reconcile=False)
        original = (box.pin_id, list(box.tags), box.identity_confirmed)

        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        activate_attempt(menu)
        menu.Destroy()
        wait_for_attempt(window)

        assert (box.pin_id, box.tags, box.identity_confirmed) == original
        assert 'kept the reviewed or manually entered identity' in (
            window.GetStatusBar().GetStatusText(1))
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_discards_result_after_navigation(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Stale attempt', '', image_array=[image] * 3,
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    started = Event()
    release = Event()

    class SlowMatcher:
        has_references = True

        def suggest(self, _crop):
            started.set()
            assert release.wait(5)
            return PinMatch('pin-new', 'New Pin', 7, 0.72)

        match = suggest

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 15, 30, 30), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.update({0, 2})
        window._ScrubberFrame__detected_frames.add(2)
        window._pin_matcher = SlowMatcher()
        window.display_image(reconcile=False)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        activate_attempt(menu)
        menu.Destroy()
        assert started.wait(5)
        pending = window._pending_attempt
        assert window.goto_frame(2)
        release.set()
        pending.result(timeout=5)
        wx.Yield()
        assert box.pin_id is None and box.tags == []
    finally:
        release.set()
        window.Destroy()
        wx.Yield()
