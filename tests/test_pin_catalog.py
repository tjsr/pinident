from io import StringIO
from time import monotonic, sleep
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData, IdentityRejection
from controls.imagepanel import ImagePanel
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


def test_best_effort_suggestion_ranks_featureless_references_and_excludes_rejected_pin():
    red = np.full((90, 90, 3), (220, 30, 30), dtype=np.uint8)
    blue = np.full((90, 90, 3), (30, 30, 220), dtype=np.uint8)
    matcher = PinMatcher([
        PinReference('pin-red', 'Red Pin', red),
        PinReference('pin-blue', 'Blue Pin', blue),
    ])

    assert matcher.match(red) is None
    first = matcher.suggest(red)
    assert first is not None and first.pin_id == 'pin-red'
    assert first.confidence is not None and 0 <= first.confidence <= 1
    second = matcher.suggest(red, excluded_pin_ids=frozenset({'pin-red'}))
    assert second is not None and second.pin_id == 'pin-blue'
    assert second.confidence is not None and 0 <= second.confidence <= 1
    assert matcher.suggest(red, excluded_pin_ids=frozenset({'pin-red', 'pin-blue'})) is None


def test_ranked_replacements_return_next_twenty_unique_unrejected_pins():
    red = np.full((60, 60, 3), (220, 30, 30), dtype=np.uint8)
    references = [PinReference(f'pin-{index:02d}', f'Pin {index:02d}', red)
                  for index in range(25)]
    references.append(PinReference('pin-00', 'Pin 00', red))
    matcher = PinMatcher(references)

    first = matcher.rank_matches(red, limit=20)
    assert [match.pin_id for match in first] == [f'pin-{index:02d}' for index in range(20)]
    assert all(match.confidence == pytest.approx(1.0) for match in first)

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
            for index in range(25)])
        panel = window._ScrubberFrame__image_panel

        menu = panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        items = list(submenu.GetMenuItems())
        assert len(items) == 22
        assert items[20].IsSeparator()
        assert items[21].GetItemLabelText() == 'none of these'
        assert items[0].GetItemLabelText() == '100% Pin 00'
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, items[21].GetId()))
        menu.Destroy()

        assert len(box.identity_rejections) == 20
        assert [review.pin_id for review in box.identity_rejections] == [
            f'pin-{index:02d}' for index in range(20)]
        assert box.review_state == 'unconfirmed' and box.pin_id is None
        assert window._feedback_store.counts()['negative'] == 0
        assert len(window._feedback_store.identity_rejections_for_source(
            window._feedback_source)) == 20
        window.add_to_training_data()
        assert '20 examples indexed' in window._ScrubberFrame__tag_panel.training_status.GetLabel()
        assert '20 wrong IDs' in window._ScrubberFrame__tag_panel.training_status.GetLabel()
        stream = StringIO()
        save_boxes_to_stream(stream, window._ScrubberFrame__frame_boxes)
        assert len(load_boxes_from_stream(StringIO(stream.getvalue()))[0][0].identity_rejections) == 20

        menu = panel.make_box_context_menu(box)
        submenu = next(item.GetSubMenu() for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Replace with pin')
        items = list(submenu.GetMenuItems())
        assert len(items) == 7
        assert [item.GetItemLabelText() for item in items[:5]] == [
            f'100% Pin {index:02d}' for index in range(20, 25)]
        submenu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, items[0].GetId()))
        menu.Destroy()

        assert (box.pin_id, box.tags) == ('pin-20', ['Pin 20'])
        assert box.review_state == 'confirmed' and box.identity_confirmed
        assert window._feedback_store.counts()['positive'] == 1
        assert len(box.identity_rejections) == 20

        window.undo_action()
        assert box.pin_id is None and not box.identity_confirmed
        assert len(box.identity_rejections) == 20
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
            for index in range(21)])
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
        assert shown[1] == ['100% Pin 20', 'none of these']
        assert shown[2] == ['No more catalog matches']
        assert len(box.identity_rejections) == 21
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
        assert [review.pin_id for review in box.identity_rejections] == [
            'pin-first', 'pin-second']
    finally:
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


def test_detect_button_scores_best_available_catalog_match_for_each_candidate(tmp_path):
    app = wx.App.Get() or wx.App(False)
    db_path, known = make_catalog(tmp_path)
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    frame[40:220, 30:210] = known
    frame[40:220, 215:395] = reference_image(2)
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
        assert boxes[1].pin_id == 'pin-1'
        assert boxes[1].match_confidence is not None
        assert panel.get_box_label_text(boxes[1]).split()[0].endswith('%')
        assert panel.get_box_colour(boxes[0]) == wx.RED
        assert panel.get_box_colour(boxes[1]) == wx.RED
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
        assert matcher.seen == [(220, 30, 30), (30, 220, 30), (220, 220, 30)]
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
