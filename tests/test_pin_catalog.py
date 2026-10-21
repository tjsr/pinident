from io import StringIO
from pathlib import Path
import sqlite3
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData, IdentityRejection
from controls.imagepanel import ImagePanel
from detection_feedback import FeedbackModel, FeedbackStore, make_feedback_example
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from pin_catalog import PinCatalog, PinMatch, PinMatcher, PinReference
from pin_detection import associate_frame_boxes, identify_candidates
from scrubberframe import (load_boxes_from_stream, propagate_confirmations,
                           save_boxes_to_stream)
from videoscrubber import VideoScrubber


def reference_image(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(15, 235, (180, 180, 3), dtype=np.uint8)
    cv2.circle(image, (90, 90), 74, (220, 35, 30), 5)
    cv2.putText(image, str(seed), (45, 108), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    return image


def make_catalog(tmp_path):
    image = reference_image(1)
    path = tmp_path / 'reference.png'
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    db_path = tmp_path / 'known.sqlite3'
    catalog = PinCatalog(db_path)
    catalog.add_reference('pin-1', 'Known Pin', path, 'local test fixture', 'test use')
    return db_path, image


def wait_for_catalog(window):
    deadline = monotonic() + 5
    while window._pending_catalog and monotonic() < deadline:
        wx.Yield()
        sleep(0.005)
    assert not window._pending_catalog


@pytest.mark.gui
def test_catalog_matching_merges_overlapping_candidates_for_same_pin(tmp_path):
    class SamePinMatcher:
        has_references = True

        def match(self, _crop):
            return PinMatch('pinpanion:7', 'Mario', 12, 0.8)

    app = wx.App.Get() or wx.App(False)
    image = reference_image(7)
    window = VideoScrubber(None, 'Catalog duplicate merge', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        first = BoxData((35, 35, 80, 80), [], 'automatic')
        second = BoxData((40, 38, 78, 78), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [first, second]
        window._pin_matcher = SamePinMatcher()
        with patch.object(window, '_schedule_reconciliation'):
            assert window._schedule_catalog_match(0, image, None)
            wait_for_catalog(window)
        boxes = window._ScrubberFrame__frame_boxes[0]
        assert len(boxes) == 1
        assert boxes[0] in (first, second)
        assert boxes[0].pin_id == 'pinpanion:7'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_background_catalog_match_clears_tiny_saved_suggestion(tmp_path: Path) -> None:
    class SamePinMatcher:
        has_references = True

        def match(self, _crop: np.ndarray) -> PinMatch:
            return PinMatch('pinpanion:7', 'Mario', 12, 0.8)

    app = wx.App.Get() or wx.App(False)
    image = np.full((320, 700, 3), 145, dtype=np.uint8)
    window = VideoScrubber(None, 'Tiny catalog suggestion', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        larger = [BoxData(coords, [], 'automatic') for coords in (
            (20, 30, 65, 60), (120, 30, 100, 75), (270, 30, 135, 100))]
        tiny = BoxData((500, 30, 10, 10), ['Old guess'], 'automatic',
                       pin_id='old-id', match_confidence=0.7)
        window._ScrubberFrame__frame_boxes[0] = [*larger, tiny]
        window._pin_matcher = SamePinMatcher()
        with patch.object(window, '_schedule_reconciliation'):
            assert window._schedule_catalog_match(0, image, None)
            wait_for_catalog(window)
        assert window._ScrubberFrame__frame_boxes[0][-1] is tiny
        assert (tiny.pin_id, tiny.tags, tiny.match_confidence) == (None, [], None)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_background_catalog_abstention_clears_confirmed_provisional_name(
        tmp_path: Path) -> None:
    class AbstainingMatcher:
        has_references = True

        def suggest(self, _crop: np.ndarray) -> None:
            return None

        match = suggest

    app = wx.App.Get() or wx.App(False)
    image = reference_image(7)
    window = VideoScrubber(None, 'Catalog abstention', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        box = BoxData((20, 20, 100, 100), ['Old guess'], 'automatic',
                      review_state='confirmed', identity_confirmed=False,
                      pin_id='old-id', match_confidence=0.95)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._pin_matcher = AbstainingMatcher()
        with patch.object(window, '_schedule_reconciliation'):
            assert window._schedule_catalog_match(0, image, None)
            wait_for_catalog(window)
        assert window._ScrubberFrame__frame_boxes[0][0] is box
        assert box.review_state == 'confirmed'
        assert (box.pin_id, box.tags, box.match_confidence) == (None, [], None)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_attempt_detection_abstention_clears_old_guess_with_undo(
        tmp_path: Path) -> None:
    class AbstainingMatcher:
        has_references = True

        def suggest(self, _crop: np.ndarray) -> None:
            return None

        match = suggest

    app = wx.App.Get() or wx.App(False)
    image = reference_image(8)
    window = VideoScrubber(None, 'Attempt abstention', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        box = BoxData((20, 20, 100, 100), ['Old guess'], 'automatic',
                      pin_id='old-id', match_confidence=0.95)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._pin_matcher = AbstainingMatcher()
        window.display_image(reconcile=False)
        with patch.object(window, '_schedule_reconciliation'):
            window.attempt_detection(box)
            deadline = monotonic() + 5
            while window._pending_attempt is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert window._pending_attempt is None
        assert (box.pin_id, box.tags, box.match_confidence) == (None, [], None)
        assert 'cleared unverified identity' in window.GetStatusBar().GetStatusText(1)
        window.undo_action()
        assert (box.pin_id, box.tags, box.match_confidence) == (
            'old-id', ['Old guess'], 0.95)
    finally:
        window.Destroy()
        wx.Yield()


def test_catalog_matches_reference_and_rejects_other_pin(tmp_path):
    db_path, image = make_catalog(tmp_path)
    matcher = PinMatcher(PinCatalog(db_path).references())
    altered = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    altered = cv2.convertScaleAbs(altered, alpha=0.85, beta=12)

    match = matcher.match(altered)

    assert match is not None
    assert (match.pin_id, match.name) == ('pin-1', 'Known Pin')
    assert match.confidence is not None and 0.60 <= match.confidence <= 1.0
    assert matcher.match(reference_image(2)) is None
    assert PinCatalog(tmp_path / 'missing.sqlite3').references() == []


def test_stock_training_images_include_only_pinpanion_imports_with_provenance(tmp_path):
    catalog = PinCatalog(tmp_path / 'known.sqlite3')
    image = reference_image(7)
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    assert ok
    assert catalog.import_pinpanion_pin(
        {'id': 7, 'name': 'Catalog Seven'}, encoded.tobytes(),
        'https://pinpanion.com/imgs/seven.png')
    local = tmp_path / 'local.png'
    local.write_bytes(encoded.tobytes())
    catalog.add_reference('local:one', 'Local pin', local, 'local fixture', 'test use')
    catalog.add_reference('pinpanion:7', 'Catalog Seven', local,
                          'local extra reference', 'test use')
    with sqlite3.connect(catalog.path) as db:
        db.execute("UPDATE pins SET name = 'Local Alias' WHERE pin_id = 'pinpanion:7'")

    stock = list(catalog.stock_training_images())

    assert len(stock) == 1
    assert (stock[0].pin_id, stock[0].name, stock[0].source, stock[0].rights) == (
        'pinpanion:7', 'Local Alias',
        'https://pinpanion.com/imgs/seven.png', 'source terms unverified')
    assert stock[0].tags == ('Catalog Seven',)
    assert stock[0].image_bytes == encoded.tobytes()
    assert catalog.stock_training_state() == (1, stock[0].reference_id,
                                              len(encoded.tobytes()))


def test_featureless_colour_alone_does_not_assign_a_pin_identity() -> None:
    red = np.full((90, 90, 3), (220, 30, 30), dtype=np.uint8)
    blue = np.full((90, 90, 3), (30, 30, 220), dtype=np.uint8)
    matcher = PinMatcher([
        PinReference('pin-red', 'Red Pin', red),
        PinReference('pin-blue', 'Blue Pin', blue),
    ])

    assert matcher.match(red) is None
    assert matcher.suggest(red) is None
    assert matcher.suggest(red, excluded_pin_ids=frozenset({'pin-red'})) is None
    assert matcher.suggest(red, excluded_pin_ids=frozenset({'pin-red', 'pin-blue'})) is None
    assert [match.pin_id for match in matcher.rank_matches(red)] == ['pin-red', 'pin-blue']


def test_best_effort_uses_spatial_pattern_to_choose_between_similar_colours() -> None:
    left_dark = np.full((96, 96, 3), 210, dtype=np.uint8)
    left_dark[:, :48] = 35
    center_dark = np.full((96, 96, 3), 210, dtype=np.uint8)
    center_dark[:, 24:72] = 35
    matcher = PinMatcher([
        PinReference('pin-left', 'Left Dark', left_dark),
        PinReference('pin-center', 'Center Dark', center_dark),
    ])

    with patch.object(matcher, 'match', return_value=None):
        suggestion = matcher.suggest(left_dark)
        assert suggestion is not None and suggestion.pin_id == 'pin-left'
        assert matcher.suggest(left_dark, excluded_pin_ids=frozenset({'pin-left'})) is None
        assert matcher.suggest(np.rot90(left_dark).copy()).pin_id == 'pin-left'
    assert matcher.rank_matches(left_dark)[0].pin_id == 'pin-left'


def test_best_effort_abstains_when_distinct_pins_look_equally_close() -> None:
    pattern = np.full((96, 96, 3), 210, dtype=np.uint8)
    pattern[:, :48] = 35
    matcher = PinMatcher([
        PinReference('pin-one', 'One', pattern),
        PinReference('pin-two', 'Two', pattern.copy()),
    ])

    with patch.object(matcher, 'match', return_value=None):
        assert matcher.suggest(pattern) is None


def test_ranked_replacements_return_next_twenty_unique_unrejected_pins():
    red = np.full((60, 60, 3), (220, 30, 30), dtype=np.uint8)
    references = [PinReference(f'pin-{index:02d}', f'Pin {index:02d}', red)
                  for index in range(25)]
    references.append(PinReference('pin-00', 'Pin 00', red))
    matcher = PinMatcher(references)

    first = matcher.rank_matches(red, limit=20)
    assert [match.pin_id for match in first] == [f'pin-{index:02d}' for index in range(20)]
    assert all(match.confidence == pytest.approx(0.35) for match in first)

    following = matcher.rank_matches(
        red, limit=20, excluded_pin_ids=frozenset(match.pin_id for match in first))
    assert [match.pin_id for match in following] == [
        f'pin-{index:02d}' for index in range(20, 25)]
    assert matcher.rank_matches(
        red, excluded_pin_ids=frozenset(match.pin_id for match in first),
        excluded_names=frozenset(match.name.casefold() for match in following)) == []


@pytest.mark.gui
def test_replacement_ranking_excludes_reviewed_id_but_keeps_same_named_pin():
    app = wx.App.Get() or wx.App(False)
    red = np.full((80, 80, 3), (220, 30, 30), dtype=np.uint8)
    window = VideoScrubber(
        None, 'Duplicate names', '',
        image_array=[cv2.cvtColor(red, cv2.COLOR_RGB2BGR)], load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        review = IdentityRejection('review-a', 'pin-a', ('Twin Pin',),
                                   (10, 10, 60, 60), 0)
        box = BoxData((10, 10, 60, 60), [], 'automatic',
                      identity_rejections=(review,))
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = PinMatcher([
            PinReference('pin-a', 'Twin Pin', red),
            PinReference('pin-b', 'Twin Pin', red)])

        assert [match.pin_id for match in window.rank_pin_replacements(box)] == ['pin-b']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_replacement_menu_displays_year_and_show_when_known(tmp_path):
    app = wx.App.Get() or wx.App(False)
    red = np.full((90, 90, 3), (220, 30, 30), dtype=np.uint8)
    window = VideoScrubber(None, 'Replacement details', '',
                           image_array=[cv2.cvtColor(red, cv2.COLOR_RGB2BGR)],
                           feedback_path=tmp_path / 'feedback.sqlite3',
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 60, 60), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = PinMatcher([
            PinReference('some', 'Some pin', red, year=2021, show='West'),
            PinReference('year', 'Year only', red, year=2020),
            PinReference('local', 'Local pin', red),
        ])
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        labels = [item.GetItemLabelText() for item in submenu.GetMenuItems()
                  if not item.IsSeparator()]
        assert '35% [2021/West] Some pin' in labels
        assert '35% [2020] Year only' in labels
        assert '35% Local pin' in labels
        menu.Destroy()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_replace_with_pin_menu_rejects_one_page_then_confirms_a_later_choice(tmp_path):
    app = wx.App.Get() or wx.App(False)
    red = np.full((100, 100, 3), (220, 30, 30), dtype=np.uint8)
    window = VideoScrubber(
        None, 'Replacement choices', '',
        image_array=[cv2.cvtColor(red, cv2.COLOR_RGB2BGR)],
        feedback_path=tmp_path / 'feedback.sqlite3', load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 60, 60), ['Pin 00'], 'automatic',
                      pin_id='pin-00', match_confidence=0.90,
                      identity_confirmed=False)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = PinMatcher([
            PinReference(f'pin-{index:02d}', f'Pin {index:02d}', red)
            for index in range(15)])
        panel = window._ScrubberFrame__image_panel

        menu = panel.make_box_context_menu(box)
        assert menu.GetMenuItems()[-1].GetItemLabelText() == 'Not a pin'
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        items = list(submenu.GetMenuItems())
        assert len(items) == 12
        assert items[10].IsSeparator()
        assert items[11].GetItemLabelText() == 'none of these'
        assert items[0].GetItemLabelText() == '35% Pin 00'
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, items[11].GetId()))
        menu.Destroy()
        wait_for_catalog(window)

        assert len(box.identity_rejections) == 10
        assert [review.pin_id for review in box.identity_rejections] == [
            f'pin-{index:02d}' for index in range(10)]
        assert box.review_state == 'unconfirmed' and box.pin_id is None
        assert box.tags == [] and not box.identity_confirmed
        assert window._feedback_store.counts()['negative'] == 0
        assert len(window._feedback_store.identity_rejections_for_source(
            window._feedback_source)) == 10
        window.add_to_training_data()
        assert '10 examples indexed' in window._ScrubberFrame__tag_panel.training_status.GetLabel()
        assert '10 wrong IDs' in window._ScrubberFrame__tag_panel.training_status.GetLabel()
        stream = StringIO()
        save_boxes_to_stream(stream, window._ScrubberFrame__frame_boxes)
        assert len(load_boxes_from_stream(StringIO(stream.getvalue()))[0][0].identity_rejections) == 10

        menu = panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        items = list(submenu.GetMenuItems())
        assert len(items) == 7
        assert [item.GetItemLabelText() for item in items[:5]] == [
            f'35% Pin {index:02d}' for index in range(10, 15)]
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, items[0].GetId()))
        menu.Destroy()

        assert (box.pin_id, box.tags) == ('pin-10', ['Pin 10'])
        assert box.review_state == 'confirmed' and box.identity_confirmed
        assert window._feedback_store.counts()['positive'] == 1
        assert len(box.identity_rejections) == 10

        window.undo_action()
        assert box.pin_id is None and not box.identity_confirmed
        assert len(box.identity_rejections) == 10
        window.undo_action()
        assert box.identity_rejections == ()
        assert (box.pin_id, box.tags) == ('pin-00', ['Pin 00'])
        assert not window._feedback_store.identity_rejections_for_source(
            window._feedback_source)

        menu = panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        replacement = list(submenu.GetMenuItems())[1]
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, replacement.GetId()))
        menu.Destroy()
        assert box.pin_id == 'pin-01' and box.identity_confirmed
        assert [review.pin_id for review in box.identity_rejections] == ['pin-00']
        assert len(window._undo_history[0]) == 1
        window.undo_action()
        assert (box.pin_id, box.tags) == ('pin-00', ['Pin 00'])
        assert box.identity_rejections == ()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_none_of_these_retries_selected_area_with_next_identity(tmp_path):
    app = wx.App.Get() or wx.App(False)
    red = np.full((90, 90, 3), (220, 30, 30), dtype=np.uint8)
    window = VideoScrubber(None, 'Replacement retry', '',
                           image_array=[cv2.cvtColor(red, cv2.COLOR_RGB2BGR)],
                           feedback_path=tmp_path / 'feedback.sqlite3',
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 60, 60), ['Pin 00'], 'automatic',
                      pin_id='pin-00', match_confidence=0.90)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = PinMatcher([
            PinReference(f'pin-{index:02d}', f'Pin {index:02d}', red)
            for index in range(11)])
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        none = submenu.GetMenuItems()[-1]
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, none.GetId()))
        menu.Destroy()

        wait_for_catalog(window)
        assert [review.pin_id for review in box.identity_rejections] == [
            f'pin-{index:02d}' for index in range(10)]
        assert (box.pin_id, box.tags) == (None, [])
        assert box.review_state == 'unconfirmed' and not box.identity_confirmed
        assert ImagePanel.get_box_label_text(box) == 'Candidate'
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        assert editor is not None
        assert 'Match' not in editor._BoxTagPanelEdit__heading_text.GetLabel()
        assert 'no other pin identity match' in window.GetStatusBar().GetStatusText(1)
        assert window._feedback_store.counts()['negative'] == 0
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_none_of_these_rejects_current_suggestion_outside_colour_top_ten(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(
        None, 'Replacement includes current suggestion', '',
        image_array=[np.zeros((100, 100, 3), dtype=np.uint8)],
        feedback_path=tmp_path / 'feedback.sqlite3', load_catalog=False)

    class Matcher:
        has_references = True

        def rank_matches(self, _crop, limit=10, **_kwargs):
            return [PinMatch(f'pin-{index:02d}', f'Pin {index:02d}', 0, 0.9)
                    for index in range(limit)]

        def suggest(self, _crop, excluded_pin_ids=frozenset(), **_kwargs):
            if 'pin-current' not in excluded_pin_ids:
                return PinMatch('pin-current', 'Current Pin', 20, 0.8)
            return PinMatch('pin-next', 'Next Pin', 0, 0.5)

        def match(self, _crop, **_kwargs):
            return None

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 70, 70), ['Current Pin'], 'automatic',
                      pin_id='pin-current', match_confidence=0.8)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = Matcher()

        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        choices = [item.GetItemLabelText() for item in submenu.GetMenuItems()
                   if not item.IsSeparator()]
        assert choices[0] == '80% Current Pin'
        none = submenu.GetMenuItems()[-1]
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, none.GetId()))
        menu.Destroy()

        wait_for_catalog(window)
        assert box.pin_id == 'pin-next'
        assert 'pin-current' in {item.pin_id for item in box.identity_rejections}
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_each_full_none_of_these_page_saves_10_distinct_wrong_ids(tmp_path):
    app = wx.App.Get() or wx.App(False)
    red = np.full((100, 100, 3), (220, 30, 30), dtype=np.uint8)
    window = VideoScrubber(None, 'Repeated wrong-ID pages', '',
                           image_array=[cv2.cvtColor(red, cv2.COLOR_RGB2BGR)],
                           feedback_path=tmp_path / 'feedback.sqlite3',
                           load_catalog=False)
    try:
        window.box_data_filename = str(tmp_path / 'video.json')
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 60, 60), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = PinMatcher([
            PinReference(f'pin-{index:02d}', f'Pin {index:02d}', red)
            for index in range(20)])
        panel = window._ScrubberFrame__image_panel

        for page in (1, 2):
            menu = panel.make_box_context_menu(box)
            submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                           if item.GetItemLabelText() == 'Replace with pin')
            items = list(submenu.GetMenuItems())
            assert len([item for item in items if not item.IsSeparator()]) == 11
            none = items[-1]
            submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, none.GetId()))
            menu.Destroy()
            wait_for_catalog(window)

            stored = window._feedback_store.identity_rejections_for_source(
                window._feedback_source)
            assert len(stored) == page * 10
            assert len({item.example_id for item in stored}) == page * 10
            assert len({item.pin_id for item in stored}) == page * 10
            assert all(item.coords == box.coords and item.crop_png for item in stored)
            assert box.pin_id is None
            assert f'10 wrong-ID examples saved ({page * 10} total)' in (
                window._ScrubberFrame__tag_panel.training_status.GetLabel())

        window.add_to_training_data()
        status = window._ScrubberFrame__tag_panel.training_status.GetLabel()
        assert '20 wrong IDs' in status
        assert '0 added now; reviews saved when marked' in status
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_none_of_these_reopens_the_replacement_list_with_next_matches(tmp_path):
    app = wx.App.Get() or wx.App(False)
    red = np.full((90, 90, 3), (220, 30, 30), dtype=np.uint8)
    window = VideoScrubber(
        None, 'Replacement paging', '',
        image_array=[cv2.cvtColor(red, cv2.COLOR_RGB2BGR)],
        feedback_path=tmp_path / 'feedback.sqlite3', load_catalog=False)

    class Mouse:
        def __init__(self, pos):
            self.pos = pos

        def GetPosition(self):
            return self.pos

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 60, 60), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window._pin_matcher = PinMatcher([
            PinReference(f'pin-{index:02d}', f'Pin {index:02d}', red)
            for index in range(11)])
        panel = window._ScrubberFrame__image_panel
        x, y, width, height = panel._box_panel_rect(box)
        shown: list[list[str]] = []

        def popup(menu, *_position):
            shown.append([item.GetItemLabelText() for item in menu.GetMenuItems()
                          if not item.IsSeparator()])
            if len(shown) == 1:
                submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                               if item.GetItemLabelText() == 'Replace with pin')
                none = list(submenu.GetMenuItems())[-1]
                submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, none.GetId()))
            elif len(shown) == 2:
                none = list(menu.GetMenuItems())[-1]
                menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, none.GetId()))

        with patch.object(panel, 'PopupMenu', side_effect=popup):
            panel.on_right_up(Mouse(wx.Point(x + width // 2, y + height // 2)))
            deadline = monotonic() + 2
            while len(shown) < 3 and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)

        assert len(shown) == 3
        assert shown[1] == ['35% Pin 10', 'none of these']
        assert shown[2] == ['No more catalog matches']
        assert len(box.identity_rejections) == 11
        assert '1 wrong-ID example saved (11 total)' in (
            window._ScrubberFrame__tag_panel.training_status.GetLabel())
    finally:
        window.Destroy()
        wx.Yield()


def test_catalog_exposes_stable_id_name_picker_options(tmp_path):
    db_path, _image = make_catalog(tmp_path)
    assert PinCatalog(db_path).list_pins() == [('pin-1', 'Known Pin')]


def test_unconfirmed_catalog_guess_is_rescanned_and_scored(tmp_path):
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((260, 260, 3), dtype=np.uint8)
    frame[40:220, 40:220] = known
    provisional = BoxData((40, 40, 180, 180), ['Wrong guess'], 'automatic',
                          pin_id='pin-wrong', match_confidence=0.22)
    presence_only = BoxData(provisional.coords, [], 'automatic',
                            review_state='confirmed', identity_confirmed=False)
    manual = BoxData(provisional.coords, ['My label'], 'user')
    verified = BoxData(provisional.coords, ['Verified name'], 'automatic',
                       pin_id='pin-verified', review_state='confirmed',
                       identity_confirmed=True)
    matcher = PinMatcher(PinCatalog(db_path).references())

    identify_candidates(frame, [provisional, presence_only, manual, verified], matcher)

    for box in (provisional, presence_only):
        assert (box.pin_id, box.tags) == ('pin-1', ['Known Pin'])
        assert box.match_confidence is not None and box.match_confidence >= 0.60
        assert ImagePanel.get_box_label_text(box).startswith(
            f'{box.match_confidence:.0%} ')
    assert (manual.pin_id, manual.tags) == (None, ['My label'])
    assert (verified.pin_id, verified.tags) == ('pin-verified', ['Verified name'])


def test_confirmed_blank_manual_area_receives_scored_suggestion(tmp_path):
    db_path, known = make_catalog(tmp_path)
    box = BoxData((0, 0, 180, 180), [''], 'user',
                  review_state='confirmed', identity_confirmed=False)
    identify_candidates(known, [box], PinMatcher(PinCatalog(db_path).references()))
    assert box.pin_id == 'pin-1'
    assert box.identity_confirmed is False
    assert ImagePanel.get_box_label_text(box) == f'{box.match_confidence:.0%} Known Pin?'


def test_confirmed_unidentified_areas_are_matched_before_auto_candidates():
    image = reference_image(9)
    candidate = BoxData((0, 0, 80, 80), [], 'automatic')
    confirmed = BoxData((0, 0, 80, 80), [], 'automatic',
                        review_state='confirmed', identity_confirmed=False)
    inherited = BoxData((0, 0, 80, 80), [], 'automatic',
                        review_state='inherited', identity_confirmed=False)
    named = BoxData((0, 0, 80, 80), ['User supplied name'], 'user',
                    review_state='confirmed', identity_confirmed=False)

    class RecordingMatcher:
        def __init__(self):
            self.calls = []

        def suggest(self, crop):
            self.calls.append(('suggest', len(self.calls)))
            return PinMatch('pin-9', 'Pin 9', 0, 0.74)

        def match(self, crop):
            self.calls.append(('match', len(self.calls)))
            return None

    matcher = RecordingMatcher()
    identify_candidates(image, [candidate, inherited, confirmed, named], matcher)
    assert [kind for kind, _ in matcher.calls] == ['suggest', 'suggest', 'match']
    assert confirmed.pin_id == inherited.pin_id == 'pin-9'
    assert candidate.pin_id is None and named.tags == ['User supplied name']


def test_catalog_abstention_clears_old_guesses_but_preserves_verified_identity() -> None:
    image = reference_image(9)

    class AbstainingMatcher:
        def suggest(self, _crop: np.ndarray) -> None:
            return None

        match = suggest

    provisional = BoxData((0, 0, 80, 80), ['Old guess'], 'automatic',
                          pin_id='old-pin', match_confidence=0.95)
    present = BoxData((80, 0, 80, 80), ['Another guess'], 'automatic',
                      review_state='confirmed', pin_id='another-pin',
                      match_confidence=0.95, identity_confirmed=False)
    verified = BoxData((0, 80, 80, 80), ['Verified pin'], 'automatic',
                       review_state='confirmed', identity_confirmed=True,
                       pin_id='verified-pin', match_confidence=0.8)
    identify_candidates(image, [provisional, present, verified],
                        AbstainingMatcher(), best_effort=True)

    for box in (provisional, present):
        assert (box.pin_id, box.tags, box.match_confidence) == (None, [], None)
    assert (verified.pin_id, verified.tags, verified.match_confidence) == (
        'verified-pin', ['Verified pin'], 0.8)
    present.pin_id = 'another-pin'
    present.tags = ['Another guess']
    present.match_confidence = 0.95
    identify_candidates(image, [present], AbstainingMatcher())
    assert (present.pin_id, present.tags, present.match_confidence) == (None, [], None)


def test_reviewed_identity_model_can_identify_confirmed_pin_when_catalog_is_weak():
    image = reference_image(10)
    store = FeedbackStore()
    try:
        store.save(make_feedback_example('reviewed-pin', 'positive', image,
                                         (0, 0, 180, 180), 'video.mp4', 0,
                                         'track', 'pin-10', ('Pin 10',)))
        model = FeedbackModel(store)

        class WeakCatalog:
            def suggest(self, _crop):
                return PinMatch('pin-wrong', 'Wrong Pin', 0, 0.35)

            match = suggest

        box = BoxData((0, 0, 180, 180), [], 'user',
                      review_state='confirmed', identity_confirmed=False)
        identify_candidates(image, [box], WeakCatalog(), feedback_model=model)
        assert box.pin_id == 'pin-10' and box.tags == ['Pin 10']
        assert box.match_confidence is not None
        assert not box.identity_confirmed
    finally:
        store.close()


def test_rejected_automatic_suggestion_can_retry_reviewed_identity_model():
    image = reference_image(10)
    store = FeedbackStore()
    try:
        store.save(make_feedback_example('reviewed-pin', 'positive', image,
                                         (0, 0, 180, 180), 'video.mp4', 0,
                                         'track', 'pin-10', ('Pin 10',)))
        model = FeedbackModel(store)

        class EmptyCatalog:
            def suggest(self, _crop, **_kwargs):
                return None

            match = suggest

        box = BoxData((0, 0, 180, 180), [], 'automatic',
                      identity_rejections=(IdentityRejection(
                          'wrong-review', 'pin-first', ('First Pin',),
                          (0, 0, 180, 180), 0),))
        identify_candidates(image, [box], EmptyCatalog(), best_effort=True,
                            feedback_model=model, prefer_reviewed_identity=True)
        assert box.pin_id == 'pin-10' and box.tags == ['Pin 10']
        assert box.review_state == 'unconfirmed' and not box.identity_confirmed
    finally:
        store.close()


@pytest.mark.gui
def test_navigation_retries_confirmed_unidentified_area_even_after_prior_catalog_pass(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = reference_image(11)
    window = VideoScrubber(None, 'Retry confirmed pin', '',
                           image_array=[image, image], load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')

    class RecordingMatcher:
        has_references = True

        def __init__(self):
            self.calls = 0

        def suggest(self, _crop):
            self.calls += 1
            return PinMatch('pin-11', 'Pin 11', 0, 0.73)

        match = suggest

    try:
        window.Show()
        wx.Yield()
        box = BoxData((0, 0, 180, 180), [], 'automatic',
                      review_state='confirmed', identity_confirmed=False)
        window._ScrubberFrame__frame_boxes[1] = [box]
        window._ScrubberFrame__detected_frames.update({0, 1})
        window._ScrubberFrame__matched_frames.add(1)
        matcher = RecordingMatcher()
        window._pin_matcher = matcher
        assert window.goto_frame(1)
        deadline = monotonic() + 5
        while (box.pin_id is None and monotonic() < deadline):
            wx.Yield()
            sleep(0.005)
        assert matcher.calls >= 1
        assert box.pin_id == 'pin-11' and not box.identity_confirmed
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_detect_button_uses_reviewed_identity_when_catalog_has_no_references(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = reference_image(12)
    feedback_path = tmp_path / 'feedback.sqlite3'
    store = FeedbackStore(feedback_path)
    store.save(make_feedback_example('reviewed-pin', 'positive', image,
                                     (0, 0, 180, 180), 'review.mp4', 0,
                                     'reviewed-track', 'pin-12', ('Pin 12',)))
    store.close()
    window = VideoScrubber(None, 'Identity from reviews', '',
                           image_array=[cv2.cvtColor(image, cv2.COLOR_RGB2BGR)],
                           load_catalog=False, feedback_path=feedback_path)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((0, 0, 180, 180), [], 'user',
                      review_state='confirmed', identity_confirmed=False)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        with patch('frame_processing.detect_pin_boxes', return_value=[]):
            window.on_detect_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert box.pin_id == 'pin-12' and box.tags == ['Pin 12']
        assert box.match_confidence is not None and not box.identity_confirmed
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_drawn_box_starts_as_pin_and_retries_scored_identity_after_rejection(tmp_path):
    app = wx.App.Get() or wx.App(False)
    frame = np.zeros((260, 260, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Drawn pin suggestion', '', image_array=[frame],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')

    class SuggestingMatcher:
        has_references = True

        def suggest(self, _crop, excluded_pin_ids=frozenset(),
                    excluded_names=frozenset()):
            if 'pin-first' not in excluded_pin_ids:
                return PinMatch('pin-first', 'First Pin', 3, 0.31)
            assert excluded_names == frozenset()
            if 'pin-second' not in excluded_pin_ids:
                return PinMatch('pin-second', 'Second Pin', 2, 0.12)
            return None

        def match(self, _crop, **_kwargs):
            pytest.fail('a user-drawn pin should use the best-effort suggestion path')

    class Mouse:
        def __init__(self, x, y):
            self.position = wx.Point(x, y)

        def GetPosition(self):
            return self.position

        def Skip(self):
            pass

    def wait_for_match():
        deadline = monotonic() + 5
        while window._pending_catalog and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert not window._pending_catalog

    try:
        window._pin_matcher = SuggestingMatcher()
        window.Show()
        wx.Yield()
        panel = window._ScrubberFrame__image_panel
        origin_x, origin_y = panel.get_image_offset()
        panel.on_left_down(Mouse(origin_x + 40, origin_y + 40))
        panel.on_left_up(Mouse(origin_x + 220, origin_y + 220))
        wx.Yield()
        box = window._ScrubberFrame__frame_boxes[0][0]
        assert box.review_state == 'confirmed'
        assert not box.identity_confirmed
        assert box.feedback_label == 'positive' and box.feedback_id
        assert window._feedback_store.counts()['positive'] == 1
        wait_for_match()
        assert (box.pin_id, box.tags, box.match_confidence) == (
            'pin-first', ['First Pin'], 0.31)
        assert ImagePanel.get_box_label_text(box) == '31% First Pin?'
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        assert 'Match 31%' in editor._BoxTagPanelEdit__heading_text.GetLabel()

        menu = panel.make_box_context_menu(box)
        wrong = next(item for item in menu.GetMenuItems()
                     if item.GetItemLabelText() == 'Not this pin')
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, wrong.GetId()))
        menu.Destroy()
        wait_for_match()
        assert box.review_state == 'confirmed' and not box.identity_confirmed
        assert (box.pin_id, box.tags, box.match_confidence) == (
            'pin-second', ['Second Pin'], 0.12)
        assert ImagePanel.get_box_label_text(box) == '12% Second Pin?'
        assert 'Match 12%' in editor._BoxTagPanelEdit__heading_text.GetLabel()
        assert [review.pin_id for review in box.identity_rejections] == ['pin-first']
        assert window._feedback_store.counts()['positive'] == 1
        assert [item.pin_id for item in
                window._feedback_store.identity_rejections_for_source(
                    window._feedback_source)] == ['pin-first']

        menu = panel.make_box_context_menu(box)
        wrong = next(item for item in menu.GetMenuItems()
                     if item.GetItemLabelText() == 'Not this pin')
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, wrong.GetId()))
        menu.Destroy()
        wait_for_match()
        assert box.pin_id is None and box.tags == []
        assert 'no other pin identity match' in window.GetStatusBar().GetStatusText(1)
        assert [review.pin_id for review in box.identity_rejections] == [
            'pin-first', 'pin-second']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('source', ['automatic', 'processed'])
def test_not_this_pin_retries_automatic_area_with_next_suggestion(tmp_path, source):
    app = wx.App.Get() or wx.App(False)
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Automatic pin retry', '', image_array=[frame],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')

    class SuggestingMatcher:
        has_references = True

        def suggest(self, _crop, excluded_pin_ids=frozenset(),
                    excluded_names=frozenset()):
            assert excluded_names == frozenset()
            if 'pin-first' not in excluded_pin_ids:
                return PinMatch('pin-first', 'First Pin', 3, 0.31)
            return PinMatch('pin-second', 'Second Pin', 2, 0.12)

        def match(self, _crop, **_kwargs):
            return None

    try:
        window._pin_matcher = SuggestingMatcher()
        window.Show()
        wx.Yield()
        box = BoxData((15, 15, 80, 80), ['First Pin'], source,
                      pin_id='pin-first', match_confidence=0.31)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)

        panel = window._ScrubberFrame__image_panel
        menu = panel.make_box_context_menu(box)
        wrong = next(item for item in menu.GetMenuItems()
                     if item.GetItemLabelText() == 'Not this pin')
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, wrong.GetId()))
        menu.Destroy()

        deadline = monotonic() + 5
        while window._pending_catalog and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert not window._pending_catalog
        # Allow the saved-frame reconciliation queued by the retry to commit.
        deadline = monotonic() + 2
        while window._pending_reconcile is not None and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert window._pending_reconcile is None
        assert box.review_state == 'unconfirmed'
        assert (box.pin_id, box.tags, box.match_confidence) == (
            'pin-second', ['Second Pin'], 0.12)
        assert 'suggested Second Pin' in window.GetStatusBar().GetStatusText(1)
        assert [review.pin_id for review in box.identity_rejections] == ['pin-first']
        assert window._feedback_store.counts()['positive'] == 0
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_not_this_pin_retry_survives_catalog_refresh(tmp_path):
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Retry after catalog refresh', '',
                           image_array=[image], load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    started = Event()
    release = Event()

    class Matcher:
        has_references = True

        def __init__(self, wait=False):
            self.wait = wait

        def suggest(self, _crop, excluded_pin_ids=frozenset(),
                    excluded_names=frozenset()):
            assert 'pin-first' in excluded_pin_ids
            if self.wait:
                started.set()
                assert release.wait(5)
            return PinMatch('pin-second', 'Second Pin', 0, 0.72)

        def match(self, _crop, **_kwargs):
            return None

    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 80, 80), ['First Pin'], 'automatic',
                      pin_id='pin-first', match_confidence=0.3)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = Matcher(wait=True)
        window.display_image(reconcile=False)

        window.mark_wrong_identity(box)
        assert started.wait(5)
        window.set_pin_matcher(Matcher())
        release.set()
        wait_for_catalog(window)

        assert (box.pin_id, box.tags) == ('pin-second', ['Second Pin'])
        assert 'suggested Second Pin' in window.GetStatusBar().GetStatusText(1)
    finally:
        release.set()
        window.pause_playback()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_not_this_pin_reports_when_no_identity_sources_are_loaded(tmp_path):
    app = wx.App.Get() or wx.App(False)
    window = VideoScrubber(None, 'No retry references', '',
                           image_array=[np.zeros((120, 120, 3), dtype=np.uint8)],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 80, 80), ['First Pin'], 'automatic',
                      pin_id='pin-first')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)

        window.mark_wrong_identity(box)

        assert box.pin_id is None and box.tags == []
        assert 'no catalog images or reviewed pin identities' in (
            window.GetStatusBar().GetStatusText(1))
    finally:
        window.pause_playback()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_newly_drawn_pin_suggestion_does_not_wait_for_other_boxes(tmp_path):
    app = wx.App.Get() or wx.App(False)
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Draw priority', '', image_array=[frame],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    other_started = Event()
    release_other = Event()

    class Matcher:
        has_references = True

        def suggest(self, crop):
            if crop.shape[:2] == (25, 25):
                return PinMatch('new-pin', 'New Pin', 3, 0.72)
            other_started.set()
            assert release_other.wait(5)
            return None

        match = suggest

    try:
        window.Show()
        wx.Yield()
        other = BoxData((60, 10, 30, 30), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [other]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = Matcher()
        window.display_image(reconcile=False)
        window._ScrubberFrame__image_panel.add_new_box((10, 10, 25, 25))
        deadline = monotonic() + 5
        while not other_started.is_set() and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert other_started.is_set()
        drawn = window._ScrubberFrame__frame_boxes[0][-1]
        assert (drawn.review_state, drawn.pin_id, drawn.tags,
                drawn.identity_confirmed) == (
            'confirmed', 'new-pin', ['New Pin'], False)
        assert other.pin_id is None
    finally:
        release_other.set()
        deadline = monotonic() + 5
        while window._pending_catalog and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        window.Destroy()
        wx.Yield()


def test_wrong_identity_exclusion_follows_tracked_area():
    from boxdata import IdentityRejection

    image = reference_image(4)
    wrong = IdentityRejection('review-1', 'pin-wrong', ('Wrong Pin',),
                              (0, 0, 180, 180), 0)
    source = BoxData((0, 0, 180, 180), [], 'automatic',
                     identity_rejections=(wrong,))
    tracked = associate_frame_boxes(image, image.copy(), [source], [source.coords])[0]
    assert tracked.identity_rejections == (wrong,)

    class RejectAwareMatcher:
        def match(self, _crop, excluded_pin_ids=frozenset(), excluded_names=frozenset()):
            assert excluded_pin_ids == frozenset({'pin-wrong'})
            return PinMatch('pin-other', 'Other Pin', 12, 0.70)

    identify_candidates(image, [tracked], RejectAwareMatcher())
    assert tracked.pin_id == 'pin-other'


def test_wrong_identity_review_clears_same_tracked_guess():
    from boxdata import IdentityRejection

    review = IdentityRejection('review-2', 'pin-wrong', ('Wrong Pin',),
                               (10, 10, 40, 40), 0)
    reviewed = BoxData((10, 10, 40, 40), [], 'automatic',
                       track_id='shared', identity_rejections=(review,))
    adjacent = BoxData((11, 10, 40, 40), ['Wrong Pin'], 'automatic',
                       track_id='shared', pin_id='pin-wrong', match_confidence=0.8)
    propagate_confirmations({0: [reviewed], 1: [adjacent]})
    assert adjacent.identity_rejections == (review,)
    assert adjacent.pin_id is None and adjacent.tags == []
    assert adjacent.match_confidence is None


@pytest.mark.gui
def test_not_this_pin_clears_verified_identity_and_green_colour(tmp_path):
    app = wx.App.Get() or wx.App(False)
    frame = reference_image(3)
    window = VideoScrubber(
        None, 'Reject verified identity', '',
        image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
        load_catalog=False, feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        window.Show()
        wx.Yield()
        box = BoxData((0, 0, 180, 180), ['Wrong Pin'], 'automatic',
                      pin_id='pin-wrong', match_confidence=0.8)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image(reconcile=False)
        window.confirm_pin_identity(box)
        assert box.review_state == 'confirmed' and box.identity_confirmed
        assert ImagePanel.get_box_colour(box) == wx.Colour(0, 160, 60)

        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        wrong = next(item for item in menu.GetMenuItems()
                     if item.GetItemLabelText() == 'Not this pin')
        with patch.object(window, '_schedule_catalog_match', return_value=False):
            menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, wrong.GetId()))
        menu.Destroy()

        assert box.review_state == 'confirmed'  # The pin-presence review remains.
        assert not box.identity_confirmed and not box.identity_inferred
        assert box.pin_id is None and box.tags == []
        assert ImagePanel.get_box_colour(box) == wx.Colour(230, 140, 0)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        confirm = next(item for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Confirm is a pin')
        assert not confirm.IsEnabled()  # The direct pin-presence review remains.
        menu.Destroy()
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        assert editor._BoxTagPanelEdit__heading_text.GetLabel().startswith(
            'Identity needs verification')
        assert window._feedback_store.counts()['positive'] == 1
        assert window._feedback_store.examples()[0].pin_id is None

        window.undo_action()
        assert box.identity_confirmed and box.pin_id == 'pin-wrong'
        assert ImagePanel.get_box_colour(box) == wx.Colour(0, 160, 60)
        window.redo_action()
        assert not box.identity_confirmed
        assert ImagePanel.get_box_colour(box) == wx.Colour(230, 140, 0)

        box.pin_id = 'pin-next'
        box.tags = ['Next Pin']
        assert ImagePanel.get_box_colour(box) == wx.Colour(230, 140, 0)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_wrong_identity_review_keeps_pin_and_indexes_crop_then_rematches(tmp_path):
    app = wx.App.Get() or wx.App(False)
    frame = reference_image(3)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    window = VideoScrubber(None, 'Wrong identity', '', image_array=[bgr],
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    class AlternateMatcher:
        has_references = True

        def match(self, _crop, excluded_pin_ids=frozenset(), excluded_names=frozenset()):
            assert 'pin-wrong' in excluded_pin_ids
            return PinMatch('pin-correct', 'Correct Pin', 14, 0.75)

    try:
        window.Show()
        wx.Yield()
        box = BoxData((0, 0, 180, 180), ['Wrong Pin'], 'automatic',
                      pin_id='pin-wrong', match_confidence=0.68,
                      review_state='confirmed', identity_confirmed=False)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window._pin_matcher = AlternateMatcher()
        window.display_image(reconcile=False)
        menu = window._ScrubberFrame__image_panel.make_box_context_menu(box)
        wrong = next(item for item in menu.GetMenuItems()
                     if item.GetItemLabelText() == 'Not this pin')
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, wrong.GetId()))
        menu.Destroy()
        assert box.review_state == 'confirmed'
        assert not box.identity_confirmed
        assert len(box.identity_rejections) == 1
        example = window._feedback_store.identity_rejections_for_source(
            window._feedback_source)[0]
        assert (example.label, example.pin_id, example.tags, example.coords) == (
            'wrong_identity', 'pin-wrong', ('Wrong Pin',), box.coords)
        assert window._feedback_store.counts() == {'positive': 0, 'negative': 0}
        deadline = monotonic() + 5
        while window._pending_catalog and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert (box.pin_id, box.tags) == ('pin-correct', ['Correct Pin'])
        assert ImagePanel.get_box_label_text(box) == '75% Correct Pin?'
        saved = StringIO()
        save_boxes_to_stream(saved, {0: [box]})
        loaded = load_boxes_from_stream(StringIO(saved.getvalue()))[0][0]
        assert loaded.identity_rejections == box.identity_rejections
        window.undo_action()
        assert box.pin_id == 'pin-wrong' and box.tags == ['Wrong Pin']
        assert not box.identity_rejections
        assert not window._feedback_store.identity_rejections_for_source(window._feedback_source)
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.parametrize('review_state', ['unconfirmed', 'confirmed'])
def test_revisiting_saved_provisional_match_scans_catalog_in_worker(tmp_path, review_state):
    app = wx.App.Get() or wx.App(False)
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((260, 260, 3), dtype=np.uint8)
    frame[40:220, 40:220] = known
    window = VideoScrubber(None, 'Saved match', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           catalog_path=str(db_path))
    try:
        window.Show()
        wx.Yield()
        box = BoxData((40, 40, 180, 180), ['Wrong guess'], 'automatic',
                      pin_id='pin-wrong', match_confidence=0.20,
                      review_state=review_state, identity_confirmed=False)
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.clear()
        window.display_image(reconcile=False)
        future = window._pending_catalog.get(0)
        assert future is not None
        assert future.result(timeout=5)[0].pin_id == 'pin-1'
        deadline = monotonic() + 5
        while window._pending_catalog and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert not window._pending_catalog
        assert (box.pin_id, box.tags) == ('pin-1', ['Known Pin'])
        assert ImagePanel.get_box_label_text(box).startswith(
            f'{box.match_confidence:.0%} ')
        assert box.identity_confirmed is False
        tag_panel = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        assert f'Match {box.match_confidence:.0%}' in tag_panel._BoxTagPanelEdit__heading_text.GetLabel()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_confirming_blank_manual_box_rescans_and_shows_match_percentage(tmp_path):
    app = wx.App.Get() or wx.App(False)
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((260, 260, 3), dtype=np.uint8)
    frame[40:220, 40:220] = known
    window = VideoScrubber(None, 'Presence then match', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           catalog_path=str(db_path))
    try:
        window.Show()
        wx.Yield()
        box = BoxData((40, 40, 180, 180), [''], 'user')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)
        window.confirm_box(box)
        deadline = monotonic() + 5
        while window._pending_catalog and monotonic() < deadline:
            wx.Yield()
            sleep(0.005)
        assert box.review_state == 'confirmed' and not box.identity_confirmed
        assert box.pin_id == 'pin-1'
        assert ImagePanel.get_box_label_text(box).startswith(f'{box.match_confidence:.0%} ')
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        assert f'Match {box.match_confidence:.0%}' in editor._BoxTagPanelEdit__heading_text.GetLabel()
    finally:
        window.Destroy()
        wx.Yield()


def test_ambiguous_catalog_images_remain_unidentified(tmp_path):
    db_path, image = make_catalog(tmp_path)
    PinCatalog(db_path).add_reference('pin-2', 'Another Pin', tmp_path / 'reference.png',
                                      'local test fixture', 'test use')
    assert PinMatcher(PinCatalog(db_path).references()).match(image) is None


def test_large_catalog_index_shortlists_correct_pin():
    references = [PinReference(f'pin-{index}', f'Pin {index}', reference_image(index))
                  for index in range(40)]
    matcher = PinMatcher(references)
    assert matcher.match(reference_image(17)).pin_id == 'pin-17'
    changed = cv2.convertScaleAbs(reference_image(17), alpha=0.8, beta=10)
    changed = cv2.warpAffine(changed, cv2.getRotationMatrix2D((90, 90), 6, 0.9),
                            (180, 180), borderMode=cv2.BORDER_REFLECT)
    assert matcher.match(changed).pin_id == 'pin-17'


def test_identification_persists_and_tracks_canonical_pin_id(tmp_path):
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[40:220, 30:210] = known
    frame[40:220, 215:395] = reference_image(2)
    known_box = BoxData((30, 40, 180, 180), [''], 'automatic')
    unknown_box = BoxData((215, 40, 180, 180), [], 'automatic')

    boxes = identify_candidates(frame, [known_box, unknown_box], PinMatcher(PinCatalog(db_path).references()))

    assert boxes[0] is known_box
    assert (known_box.pin_id, known_box.tags) == ('pin-1', ['Known Pin'])
    assert known_box.match_confidence is not None
    assert (unknown_box.pin_id, unknown_box.tags) == (None, [])
    stream = StringIO()
    save_boxes_to_stream(stream, {0: boxes})
    stream.seek(0)
    restored = load_boxes_from_stream(stream)[0]
    assert restored[0].pin_id == 'pin-1'
    assert restored[0].match_confidence == known_box.match_confidence
    assert restored[1].pin_id is None
    tracked = associate_frame_boxes(frame, frame.copy(), [known_box], [known_box.coords])
    assert tracked[0].pin_id == 'pin-1'
    assert tracked[0].tags == ['Known Pin']


def test_detect_button_leaves_visually_unmatched_candidate_unidentified(
        tmp_path: Path) -> None:
    app = wx.App.Get() or wx.App(False)
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[40:220, 30:210] = known
    frame[40:220, 215:395] = (20, 30, 220)
    window = VideoScrubber(None, 'Catalog test', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           catalog_path=str(db_path))
    try:
        window.Show()
        wx.Yield()
        with patch('scrubberframe.detect_pin_boxes', return_value=[(30, 40, 180, 180), (215, 40, 180, 180)]):
            window._ScrubberFrame__on_process(None)
        boxes = window._ScrubberFrame__frame_boxes[0]
        panel = window._ScrubberFrame__image_panel

        assert boxes[0].pin_id == 'pin-1'
        assert boxes[0].tags == ['Known Pin']
        assert panel.get_box_label_text(boxes[0]).split()[0].endswith('%')
        assert boxes[1].pin_id is None
        assert boxes[1].match_confidence is None
        assert panel.get_box_colour(boxes[0]) == wx.RED
        assert panel.get_box_colour(boxes[1]) == wx.BLUE
        assert window._ScrubberFrame__tag_panel.boxes == boxes

        tag_panel = window._ScrubberFrame__tag_panel.find_panel_for_box(boxes[0])
        tag_panel._BoxTagPanelEdit__on_label_edited(BoxLabelEditedEvent(tag_panel, 0, 'Corrected'))
        assert boxes[0].tags == ['Corrected']
        assert boxes[0].pin_id is None
        assert boxes[0].match_confidence is None
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_detect_button_matches_existing_candidate_and_presence_box_without_touching_verified_pin():
    app = wx.App.Get() or wx.App(False)
    frame = np.zeros((100, 290, 3), dtype=np.uint8)
    frame[10:70, 10:70] = (220, 30, 30)
    frame[10:70, 80:140] = (30, 220, 30)
    frame[10:70, 150:210] = (30, 30, 220)
    frame[10:70, 220:280] = (220, 220, 30)
    window = VideoScrubber(None, 'Detect existing boxes', '',
                           image_array=[cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)],
                           load_catalog=False)

    class SuggestingMatcher:
        has_references = True

        def __init__(self):
            self.seen = []

        def suggest(self, crop, **_kwargs):
            colour = tuple(int(value) for value in crop[0, 0])
            self.seen.append(colour)
            if colour == (220, 30, 30):
                return PinMatch('pin-red', 'Red Pin', 2, 0.42)
            if colour == (30, 220, 30):
                return PinMatch('pin-green', 'Green Pin', 3, 0.57)
            if colour == (220, 220, 30):
                return PinMatch('pin-yellow', 'Yellow Pin', 2, 0.33)
            pytest.fail('verified identity was sent to the matcher')

        def match(self, _crop, **_kwargs):
            return None

    try:
        window.Show()
        wx.Yield()
        candidate = BoxData((10, 10, 60, 60), [], 'automatic')
        presence = BoxData((80, 10, 60, 60), [], 'automatic',
                           review_state='confirmed', identity_confirmed=False)
        verified = BoxData((150, 10, 60, 60), ['Verified Pin'], 'automatic',
                           pin_id='pin-verified', review_state='confirmed',
                           identity_confirmed=True)
        manual_presence = BoxData((220, 10, 60, 60), [], 'user',
                                  review_state='confirmed', identity_confirmed=False)
        window._ScrubberFrame__frame_boxes[0] = [
            candidate, presence, verified, manual_presence]
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)
        matcher = SuggestingMatcher()
        window._pin_matcher = matcher

        # A fresh detector hit inside an already verified pin must not create
        # another candidate or send that region back through catalog matching.
        duplicate_verified = (160, 20, 30, 30)
        with patch('frame_processing.detect_pin_boxes', return_value=[duplicate_verified]):
            window.on_detect_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)

        assert window._pending_frame_job is None
        assert len(window._ScrubberFrame__frame_boxes[0]) == 4
        assert (candidate.pin_id, candidate.tags, candidate.match_confidence) == (
            'pin-red', ['Red Pin'], 0.42)
        assert (presence.pin_id, presence.tags, presence.match_confidence) == (
            'pin-green', ['Green Pin'], 0.57)
        assert presence.review_state == 'confirmed' and not presence.identity_confirmed
        assert (verified.pin_id, verified.tags, verified.match_confidence) == (
            'pin-verified', ['Verified Pin'], None)
        assert verified.identity_confirmed and verified.review_state == 'confirmed'
        assert (manual_presence.pin_id, manual_presence.tags,
                manual_presence.match_confidence) == (
                    'pin-yellow', ['Yellow Pin'], 0.33)
        assert matcher.seen == [(220, 220, 30), (30, 220, 30), (220, 30, 30)]
        assert ImagePanel.get_box_label_text(candidate) == '42% Red Pin'
        assert ImagePanel.get_box_label_text(presence) == '57% Green Pin?'
        assert ImagePanel.get_box_label_text(manual_presence) == '33% Yellow Pin?'
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(presence)
        assert 'Match 57%' in editor._BoxTagPanelEdit__heading_text.GetLabel()
    finally:
        window.Destroy()
        wx.Yield()


def test_programmatic_label_refresh_keeps_confirmed_catalog_id():
    app = wx.App.Get() or wx.App(False)
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Label refresh', '', image_array=[frame],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((10, 10, 40, 40), ['Old suggestion'], 'automatic',
                      pin_id='pinpanion:7', review_state='confirmed')
        window._ScrubberFrame__frame_boxes[0] = [box]
        panel = window._ScrubberFrame__tag_panel
        panel.boxes = [box]
        box.tags = ['Confirmed pin name']
        panel.boxes = [box]
        deadline = monotonic() + 0.65
        while monotonic() < deadline:
            wx.Yield()
            sleep(0.01)
        assert box.pin_id == 'pinpanion:7'
        assert box.review_state == 'confirmed'
        assert box.tags == ['Confirmed pin name']
    finally:
        window.Destroy()
        wx.Yield()


def test_next_frame_identifies_new_catalog_pin(tmp_path):
    app = wx.App.Get() or wx.App(False)
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[40:220, 30:210] = known
    frames = [np.zeros_like(frame), frame]
    window = VideoScrubber(None, 'Catalog tracking test', '',
                           image_array=[cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in frames],
                           catalog_path=str(db_path))
    try:
        window.Show()
        wx.Yield()
        with patch('scrubberframe.detect_pin_boxes', return_value=[(30, 40, 180, 180)]):
            window.on_next(None)
        box = window._ScrubberFrame__frame_boxes[1][0]
        assert (box.pin_id, box.tags) == ('pin-1', ['Known Pin'])
        assert window._ScrubberFrame__image_panel.boxes[0] is box
    finally:
        window.Destroy()
        wx.Yield()
