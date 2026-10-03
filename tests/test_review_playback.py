from io import StringIO
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
import pytest
import wx

from boxdata import BoxData
from controls.imagepanel import ImagePanel
from frame_processing import FrameJob, process_frame
from pin_catalog import PinMatch
from pin_detection import (add_new_detections, associate_frame_boxes,
                           identify_candidates, match_existing_candidates)
from scrubberframe import (load_boxes_from_stream, propagate_confirmations,
                           reconcile_saved_frames, save_boxes_to_stream)
from videoscrubber import VideoScrubber


def wait_for_reconciliation(window):
    future = window._pending_reconcile
    if future is None:
        return
    future.result(timeout=5)
    deadline = monotonic() + 5
    while window._pending_reconcile is not None and monotonic() < deadline:
        wx.Yield()
        sleep(0.005)
    assert window._pending_reconcile is None


def test_confirmation_persists_and_propagates_both_directions():
    early = BoxData((10, 10, 30, 30), [], 'automatic', track_id='same')
    anchor = BoxData((11, 10, 30, 30), ['Mario'], 'automatic', track_id='same',
                     review_state='confirmed')
    later = BoxData((12, 10, 30, 30), [], 'automatic', track_id='same')
    unrelated = BoxData((40, 40, 20, 20), [], 'automatic', track_id='other')
    frames = {0: [early], 1: [anchor, unrelated], 2: [later]}

    propagate_confirmations(frames)
    assert [early.review_state, anchor.review_state, later.review_state,
            unrelated.review_state] == ['inherited', 'confirmed', 'inherited', 'unconfirmed']

    output = StringIO()
    save_boxes_to_stream(output, frames)
    output.seek(0)
    loaded = load_boxes_from_stream(output)
    assert [loaded[index][0].review_state for index in (0, 1, 2)] == [
        'inherited', 'confirmed', 'inherited']
    assert load_boxes_from_stream(StringIO(
        '{"0": [{"coords": [1, 2, 3, 4], "tags": []}]}'))[0][0].review_state == 'unconfirmed'

    frames[1].remove(anchor)
    propagate_confirmations(frames)
    assert early.review_state == later.review_state == 'unconfirmed'


@pytest.mark.gui
def test_inherited_pin_can_be_confirmed_in_current_frame_without_changing_earlier_review():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((90, 120, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Direct frame review', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        earlier = BoxData((10, 10, 35, 35), ['Mario'], 'automatic',
                          track_id='same', review_state='confirmed',
                          pin_id='pinpanion:7', identity_confirmed=True)
        current = BoxData((10, 10, 35, 35), ['Mario'], 'automatic',
                          track_id='same', review_state='inherited',
                          pin_id='pinpanion:7', identity_confirmed=True)
        window._ScrubberFrame__frame_boxes.update({0: [earlier], 1: [current]})
        assert window.goto_frame(1)
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(current)
        assert 'another frame' in editor._BoxTagPanelEdit__heading_text.GetLabel()
        assert editor.confirm_button.IsEnabled()

        editor.confirm_button.ProcessEvent(wx.CommandEvent(
            wx.EVT_BUTTON.typeId, editor.confirm_button.GetId()))
        wait_for_reconciliation(window)
        assert current.review_state == 'confirmed'
        assert 'another frame' not in editor._BoxTagPanelEdit__heading_text.GetLabel()
        assert earlier.review_state == 'confirmed' and earlier.identity_confirmed
        assert not editor.confirm_button.IsEnabled()

        assert editor.confirm_identity_button.IsEnabled()
        editor.confirm_identity_button.ProcessEvent(wx.CommandEvent(
            wx.EVT_BUTTON.typeId, editor.confirm_identity_button.GetId()))
        wait_for_reconciliation(window)
        assert current.identity_confirmed and current.review_state == 'confirmed'
        assert not current.identity_inferred
        assert not editor.confirm_identity_button.IsEnabled()
        assert earlier.identity_confirmed and earlier.review_state == 'confirmed'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.parametrize('identity_confirmed', [False, True])
def test_adjacent_direct_confirmations_link_without_losing_either_review(identity_confirmed):
    image = np.random.default_rng(111).integers(0, 255, (90, 120, 3), dtype=np.uint8)
    name = ['Mario'] if identity_confirmed else []
    pin_id = 'pinpanion:7' if identity_confirmed else None
    earlier = BoxData((10, 10, 35, 35), name.copy(), 'automatic',
                      track_id='earlier', pin_id=pin_id, review_state='confirmed',
                      identity_confirmed=identity_confirmed)
    current = BoxData((10, 10, 35, 35), name.copy(), 'automatic',
                      track_id='current', pin_id=pin_id, review_state='confirmed',
                      identity_confirmed=identity_confirmed)
    later = BoxData((10, 10, 35, 35), name.copy(), 'automatic',
                    track_id='later', pin_id=pin_id, review_state='confirmed',
                    identity_confirmed=identity_confirmed)
    frames = {0: [earlier], 1: [current], 2: [later]}

    assert reconcile_saved_frames(1, 3, frames, lambda _index: image, reference_index=0)
    assert earlier.track_id == current.track_id == later.track_id
    assert earlier.review_state == current.review_state == later.review_state == 'confirmed'
    assert (earlier.identity_confirmed == current.identity_confirmed ==
            later.identity_confirmed == identity_confirmed)


def test_adjacent_conflicting_direct_pin_ids_remain_separate():
    image = np.random.default_rng(112).integers(0, 255, (90, 120, 3), dtype=np.uint8)
    earlier = BoxData((10, 10, 35, 35), ['Mario'], 'automatic', track_id='earlier',
                      pin_id='pinpanion:7', review_state='confirmed', identity_confirmed=True)
    current = BoxData((10, 10, 35, 35), ['Luigi'], 'automatic', track_id='current',
                      pin_id='pinpanion:8', review_state='confirmed', identity_confirmed=True)
    frames = {0: [earlier], 1: [current]}

    assert reconcile_saved_frames(1, 2, frames, lambda _index: image, reference_index=0)
    assert earlier.track_id == 'earlier' and current.track_id == 'current'
    assert earlier.tags == ['Mario'] and current.tags == ['Luigi']
    assert earlier.identity_confirmed and current.identity_confirmed


def test_confirmed_unknown_track_skips_catalog_matching():
    image = np.random.default_rng(1).integers(0, 255, (90, 90, 3), dtype=np.uint8)
    confirmed = BoxData((10, 10, 35, 35), [], 'automatic', review_state='confirmed')
    tracked = associate_frame_boxes(image, image.copy(), [confirmed], [confirmed.coords])[0]
    assert tracked.review_state == 'inherited'
    matcher = type('Matcher', (), {'match': lambda self, crop: pytest.fail('matched confirmed track')})()
    identify_candidates(image, [tracked], matcher)


def test_existing_candidate_match_requires_the_same_appearance():
    previous = np.random.default_rng(21).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    current = previous.copy()
    source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed')
    same = BoxData((10, 15, 40, 40), [], 'automatic')
    other = BoxData((110, 15, 40, 40), [], 'automatic')

    assert match_existing_candidates(previous, current, [source], [same, other]) == [(source, same)]

    manually_labelled = BoxData(same.coords, ['My correction'], 'automatic')
    assert match_existing_candidates(previous, current, [source], [manually_labelled]) == []

    current[15:55, 10:50] = np.random.default_rng(22).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    assert match_existing_candidates(previous, current, [source], [same, other]) == []


def test_saved_model_guess_can_be_linked_to_confirmed_identity():
    image = np.random.default_rng(24).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    confirmed = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                        pin_id='pinpanion:7', review_state='confirmed')
    provisional = BoxData((10, 15, 40, 40), ['Wrong match'], 'automatic',
                          pin_id='pinpanion:99')
    assert match_existing_candidates(image, image, [confirmed], [provisional]) == [
        (confirmed, provisional)]


@pytest.mark.parametrize('target_source', ['automatic', 'user'])
def test_matching_confirmed_identity_accepts_the_same_saved_name(target_source):
    image = np.random.default_rng(124).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    confirmed = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                        pin_id='pinpanion:7', review_state='confirmed')
    same_name = BoxData(confirmed.coords, [' mario '], target_source)
    assert match_existing_candidates(image, image, [confirmed], [same_name]) == [
        (confirmed, same_name)]
    same_name.tags = ['Luigi']
    assert match_existing_candidates(image, image, [confirmed], [same_name]) == []


def test_confirmed_catalog_identity_overrides_saved_guesses_across_multiple_hops():
    image = np.random.default_rng(84).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    confirmed = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                        pin_id='pinpanion:7', review_state='confirmed')

    class NoRematch:
        def match(self, _crop):
            pytest.fail('a tracked confirmed identity was sent to the catalog matcher')

    previous = confirmed
    with patch('frame_processing.detect_pin_boxes', return_value=[]):
        for index in (1, 2, 3):
            provisional = BoxData((10, 15, 40, 40), ['Wrong match'], 'automatic',
                                  pin_id=f'pinpanion:{index + 90}')
            job = FrameJob(index, index - 1, [previous], [provisional], [],
                           image, image, None, NoRematch())
            result = process_frame(job, lambda _index: pytest.fail('decoded a cached frame'))
            tracked = result.boxes[0]
            assert tracked is provisional
            assert tracked.track_id == confirmed.track_id
            assert tracked.review_state == 'inherited'
            assert tracked.pin_id == 'pinpanion:7'
            assert tracked.tags == ['Mario']
            previous = tracked


def test_new_detection_inherits_verified_identity_when_catalog_and_position_agree():
    previous = np.random.default_rng(260).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    current = np.random.default_rng(261).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=True)

    class SamePinMatcher:
        def match(self, _crop):
            return PinMatch('pinpanion:7', 'Mario', 12, 0.74)

    job = FrameJob(1, 0, [source], [], [], current, previous, None, SamePinMatcher())
    with patch('frame_processing.detect_pin_boxes', return_value=[source.coords]):
        result = process_frame(job, lambda _index: pytest.fail('decoded cached frame'))
    tracked = result.boxes[0]
    assert tracked.track_id == source.track_id
    assert tracked.review_state == 'inherited'
    assert tracked.identity_confirmed
    assert tracked.pin_id == source.pin_id
    assert ImagePanel.get_box_colour(tracked) == wx.Colour(230, 140, 0)


def test_catalog_agreement_does_not_link_distant_or_unverified_identity():
    previous = np.random.default_rng(262).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    current = np.random.default_rng(263).integers(0, 255, (100, 180, 3), dtype=np.uint8)

    class SamePinMatcher:
        def match(self, _crop):
            return PinMatch('pinpanion:7', 'Mario', 12, 0.74)

    for source, coords in (
            (BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=True), (110, 15, 40, 40)),
            (BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=False), (10, 15, 40, 40))):
        job = FrameJob(1, 0, [source], [], [], current, previous, None,
                       SamePinMatcher())
        with patch('frame_processing.detect_pin_boxes', return_value=[coords]):
            tracked = process_frame(job, lambda _index: None).boxes[0]
        assert tracked.track_id != source.track_id
        assert tracked.review_state == 'unconfirmed'
        assert not tracked.identity_confirmed


def test_saved_catalog_guess_inherits_verified_identity_with_location_agreement():
    previous = np.random.default_rng(266).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    current = np.random.default_rng(267).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=True)
    saved = BoxData((11, 15, 40, 40), ['Mario'], 'automatic',
                    pin_id='pinpanion:7', match_confidence=0.72)
    frames = {0: [source], 1: [saved]}
    assert reconcile_saved_frames(1, 2, frames,
                                  lambda index: previous if index == 0 else current,
                                  reference_index=0)
    assert saved.track_id == source.track_id
    assert saved.identity_confirmed and saved.review_state == 'inherited'


@pytest.mark.gui
def test_next_button_shows_inherited_verified_identity_for_new_catalog_detection():
    app = wx.App.Get() or wx.App(False)
    previous = np.random.default_rng(264).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    current = np.random.default_rng(265).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Catalog tracking status', '',
                           image_array=[previous, current], load_catalog=False)

    class SamePinMatcher:
        has_references = True

        def match(self, _crop):
            return PinMatch('pinpanion:7', 'Mario', 12, 0.74)

    try:
        window.Show()
        wx.Yield()
        source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                         pin_id='pinpanion:7')
        window._ScrubberFrame__frame_boxes[0] = [source]
        window._ScrubberFrame__detected_frames.add(0)
        window._ScrubberFrame__matched_frames.add(0)
        window.display_image(reconcile=False)
        window.confirm_pin_identity(source)
        window._pin_matcher = SamePinMatcher()
        with patch('frame_processing.detect_pin_boxes', return_value=[source.coords]):
            window.on_next_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        tracked = window._ScrubberFrame__frame_boxes[1][0]
        assert tracked.track_id == source.track_id
        assert tracked.identity_confirmed and tracked.review_state == 'inherited'
        assert ImagePanel.get_box_colour(tracked) == wx.Colour(230, 140, 0)
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(tracked)
        assert 'Identity confirmed in another frame' in (
            editor._BoxTagPanelEdit__heading_text.GetLabel())
    finally:
        window.Destroy()
        wx.Yield()


def test_processing_saved_same_name_carries_confirmed_identity_without_rematching():
    image = np.random.default_rng(125).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    confirmed = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                        pin_id='pinpanion:7', review_state='confirmed')
    saved = BoxData(confirmed.coords, ['Mario'], 'user')
    job = FrameJob(1, 0, [confirmed], [saved], [], image, image, None,
                   type('NoRematch', (), {
                       'match': lambda self, crop: pytest.fail('rematched a confirmed identity')
                   })())
    with patch('frame_processing.detect_pin_boxes', return_value=[]):
        result = process_frame(job, lambda _index: pytest.fail('decoded a cached frame'))
    assert result.boxes[0] is saved
    assert saved.track_id == confirmed.track_id
    assert saved.review_state == 'inherited'
    assert saved.identity_confirmed
    assert saved.pin_id == 'pinpanion:7'
    assert saved.tags == ['Mario']


def test_pin_presence_confirmation_tracks_without_approving_suggested_name():
    image = np.random.default_rng(85).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    source = BoxData((10, 15, 40, 40), ['Possible Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=False)
    provisional = BoxData(source.coords, ['Possible Luigi'], 'automatic',
                          pin_id='pinpanion:8')
    job = FrameJob(1, 0, [source], [provisional], [], image, image, None,
                   type('NoRematch', (), {
                       'match': lambda self, crop: pytest.fail('rematched a known candidate')
                   })())
    with patch('frame_processing.detect_pin_boxes', return_value=[]):
        result = process_frame(job, lambda _index: pytest.fail('decoded a cached frame'))
    tracked = result.boxes[0]
    assert tracked.track_id == source.track_id
    assert tracked.review_state == 'inherited'
    assert not tracked.identity_confirmed
    assert tracked.pin_id == 'pinpanion:8'
    assert ImagePanel.get_box_label_text(source) == 'Possible Mario?'


@pytest.mark.parametrize('target_state', ['inherited', 'confirmed'])
def test_verified_adjacent_identity_upgrades_presence_only_saved_track(target_state):
    image = np.random.default_rng(141).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    verified = BoxData((10, 15, 40, 40), ['Mario'], 'automatic', track_id='verified',
                       pin_id='pinpanion:7', review_state='confirmed',
                       identity_confirmed=True)
    presence = BoxData(verified.coords, ['Mario'], 'automatic', track_id='presence',
                       pin_id='pinpanion:7', review_state=target_state,
                       identity_confirmed=False)
    earlier = BoxData(verified.coords, ['Mario'], 'automatic', track_id='presence',
                      pin_id='pinpanion:7', review_state='inherited',
                      identity_confirmed=False)
    frames = {0: [earlier], 1: [presence], 2: [verified]}
    assert reconcile_saved_frames(1, 3, frames, lambda _index: image,
                                  reference_index=2)
    assert verified.track_id == presence.track_id == earlier.track_id
    assert presence.identity_inferred == (target_state == 'confirmed')
    assert presence.identity_confirmed and earlier.identity_confirmed
    assert presence.review_state == target_state
    assert earlier.review_state == 'inherited'
    saved = StringIO()
    save_boxes_to_stream(saved, frames)
    restored = load_boxes_from_stream(StringIO(saved.getvalue()))
    assert restored[1][0].identity_inferred == (target_state == 'confirmed')


def test_correcting_confirmed_name_withdraws_inherited_identity_trust():
    direct = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     track_id='one', pin_id='pinpanion:7', review_state='confirmed')
    linked = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     track_id='one', pin_id='pinpanion:7', review_state='inherited',
                     identity_confirmed=True)
    frames = {0: [direct], 1: [linked]}
    propagate_confirmations(frames)
    assert linked.identity_confirmed
    direct.tags = ['Corrected name']
    direct.pin_id = None
    direct.identity_confirmed = False
    propagate_confirmations(frames)
    assert not linked.identity_confirmed
    assert linked.review_state == 'inherited'
    assert ImagePanel.get_box_label_text(linked) == 'Mario?'


def test_inferred_identity_on_presence_confirmation_is_not_a_new_authority():
    direct = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     track_id='same', pin_id='pinpanion:7', review_state='confirmed',
                     identity_confirmed=True)
    inferred = BoxData(direct.coords, ['Mario'], 'automatic', track_id='same',
                       pin_id='pinpanion:7', review_state='confirmed',
                       identity_confirmed=True, identity_inferred=True)
    frames = {0: [direct], 1: [inferred]}
    direct.identity_confirmed = False
    propagate_confirmations(frames)
    assert inferred.review_state == 'confirmed'
    assert not inferred.identity_confirmed


@pytest.mark.gui
def test_previous_frame_upgrades_saved_presence_only_identity():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(142).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Presence identity link', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        presence = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                           track_id='presence', pin_id='pinpanion:7',
                           review_state='confirmed', identity_confirmed=False)
        verified = BoxData(presence.coords, ['Mario'], 'automatic',
                           track_id='verified', pin_id='pinpanion:7',
                           review_state='confirmed', identity_confirmed=True)
        window._ScrubberFrame__frame_boxes.update({0: [presence], 1: [verified]})
        window._ScrubberFrame__detected_frames.update({0, 1})
        assert window.goto_frame(1)
        wait_for_reconciliation(window)
        assert window.goto_frame(0)
        wait_for_reconciliation(window)
        assert presence.identity_confirmed and presence.identity_inferred
        assert presence.track_id == verified.track_id
        assert window._ScrubberFrame__tag_panel.find_panel_for_box(presence) is not None
    finally:
        window.Destroy()
        wx.Yield()


def test_presence_only_confirmation_of_an_inherited_box_does_not_approve_name():
    direct = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     track_id='one', pin_id='pinpanion:7', review_state='confirmed')
    linked = BoxData(direct.coords, ['Mario'], 'automatic', track_id='one',
                     pin_id='pinpanion:7', review_state='inherited',
                     identity_confirmed=True)
    linked.review_state = 'confirmed'
    linked.identity_confirmed = False
    propagate_confirmations({0: [direct], 1: [linked]})
    assert not linked.identity_confirmed
    assert ImagePanel.get_box_label_text(linked) == 'Mario?'


def test_legacy_unknown_placeholder_is_not_copied_as_a_pin_identity():
    image = np.random.default_rng(86).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    old = BoxData((10, 15, 40, 40), ['unknown-1'], 'user')
    tracked = associate_frame_boxes(image, image.copy(), [old], [old.coords])[0]
    assert tracked.tags == []
    assert not tracked.identity_confirmed
    assert ImagePanel.get_box_label_text(old) == 'Candidate'


@pytest.mark.gui
@pytest.mark.parametrize('source_state, expected_state', [
    ('confirmed', 'inherited'), ('unconfirmed', 'unconfirmed')])
def test_navigating_to_existing_candidates_links_back_through_frame_chain(source_state, expected_state):
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(23).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Existing track test', '', image_array=[image] * 3,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        early = BoxData((10, 15, 40, 40), [], 'automatic', track_id='early')
        middle = BoxData((10, 15, 40, 40), [], 'automatic', track_id='middle')
        known = BoxData((10, 15, 40, 40), ['Mario'], 'automatic', track_id='known',
                        pin_id='pinpanion:7', review_state=source_state)
        unrelated = BoxData((110, 15, 40, 40), [], 'automatic', track_id='other')
        window._ScrubberFrame__frame_boxes.update({0: [early, unrelated], 1: [middle], 2: [known]})
        window._ScrubberFrame__detected_frames.update({0, 1, 2})
        window._invalidate_reconciliation()

        assert window.goto_frame(0)
        wait_for_reconciliation(window)
        assert early.track_id == middle.track_id == known.track_id
        assert early.tags == middle.tags == ['Mario']
        assert early.pin_id == middle.pin_id == 'pinpanion:7'
        assert early.review_state == middle.review_state == expected_state
        assert unrelated.track_id == 'other' and unrelated.tags == []
        assert window._ScrubberFrame__tag_panel.find_panel_for_box(early) is not None

        output = StringIO()
        save_boxes_to_stream(output, window._ScrubberFrame__frame_boxes)
        loaded = load_boxes_from_stream(StringIO(output.getvalue()))
        assert loaded[0][0].track_id == loaded[2][0].track_id
        assert loaded[0][0].pin_id == 'pinpanion:7'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_next_frame_reconciles_saved_candidate_before_catalog_matching():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(25).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Forward existing track test', '', image_array=[image] * 3,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        known = BoxData((10, 15, 40, 40), ['Mario'], 'automatic', track_id='known',
                        pin_id='pinpanion:7', review_state='confirmed')
        saved = BoxData((10, 15, 40, 40), ['Wrong match'], 'automatic',
                        track_id='saved', pin_id='pinpanion:99')
        window._ScrubberFrame__frame_boxes.update({0: [known], 1: [saved]})
        window._ScrubberFrame__detected_frames.update({0, 1})
        window._pin_matcher = type('Matcher', (), {
            'match': lambda self, crop: pytest.fail('matched an inherited candidate')})()

        window.on_next(None)
        wait_for_reconciliation(window)
        assert saved.track_id == known.track_id
        assert saved.review_state == 'inherited'
        assert saved.pin_id == 'pinpanion:7' and saved.tags == ['Mario']

        with patch('scrubberframe.detect_pin_boxes', return_value=[(10, 15, 40, 40)]):
            window.on_next(None)
        new = window._ScrubberFrame__frame_boxes[2][0]
        assert new.track_id == known.track_id
        assert new.review_state == 'inherited'
        assert new.pin_id == 'pinpanion:7'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('target_index,target_source', [
    (0, 'automatic'), (0, 'user'), (2, 'automatic'), (2, 'user')])
def test_navigating_to_same_named_box_inherits_confirmed_identity(target_index, target_source):
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(126).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Named track test', '', image_array=[image] * 3,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        confirmed = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                            track_id='confirmed-track', pin_id='pinpanion:7',
                            review_state='confirmed')
        saved = BoxData(confirmed.coords, ['Mario'], target_source,
                        track_id='saved-track')
        window._ScrubberFrame__frame_boxes.update({1: [confirmed], target_index: [saved]})
        window._ScrubberFrame__detected_frames.update({1, target_index})
        window._invalidate_reconciliation()
        assert window.goto_frame(target_index)
        wait_for_reconciliation(window)
        assert saved.track_id == confirmed.track_id
        assert saved.review_state == 'inherited'
        assert saved.identity_confirmed
        assert saved.pin_id == 'pinpanion:7'
        assert saved.tags == ['Mario']
        assert ImagePanel.get_box_colour(saved) == wx.Colour(230, 140, 0)
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(saved)
        assert editor is not None
        assert 'Identity confirmed in another frame' in editor._BoxTagPanelEdit__heading_text.GetLabel()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_new_detection_links_a_box_already_saved_in_destination_frame():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(26).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Merged track test', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        known = BoxData((10, 15, 40, 40), ['Mario'], 'automatic', track_id='known',
                        pin_id='pinpanion:7', review_state='confirmed')
        saved = BoxData((10, 15, 40, 40), [], 'automatic', track_id='saved')
        window._ScrubberFrame__frame_boxes.update({0: [known], 1: [saved]})
        window._ScrubberFrame__detected_frames.add(0)
        window._pin_matcher = type('Matcher', (), {
            'match': lambda self, crop: pytest.fail('matched an inherited candidate')})()

        with patch('scrubberframe.detect_pin_boxes', return_value=[saved.coords]):
            window.on_next(None)
        assert window._ScrubberFrame__frame_boxes[1] == [saved]
        assert saved.track_id == known.track_id
        assert saved.tags == ['Mario'] and saved.pin_id == 'pinpanion:7'
        assert saved.review_state == 'inherited'
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_background_frame_commit_preserves_manual_name_and_inherited_identity():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(127).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Manual saved track test', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        confirmed = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                            pin_id='pinpanion:7', review_state='confirmed')
        saved = BoxData(confirmed.coords, ['Mario'], 'user')
        window._ScrubberFrame__frame_boxes.update({0: [confirmed], 1: [saved]})
        window._ScrubberFrame__detected_frames.add(0)
        window.display_image()
        with patch('frame_processing.detect_pin_boxes', return_value=[]):
            window.on_next_async(None)
            future = window._pending_frame_job
            assert future is not None
            future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert window._pending_frame_job is None
        assert window.current_index == 1
        assert window._ScrubberFrame__frame_boxes[1] == [saved]
        assert saved.track_id == confirmed.track_id
        assert saved.review_state == 'inherited' and saved.identity_confirmed
        assert saved.pin_id == 'pinpanion:7' and saved.tags == ['Mario']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('target_source', ['automatic', 'user'])
def test_playback_upgrades_saved_inherited_box_from_adjacent_verified_identity(target_source):
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(1276).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Upgrade inherited identity', '',
                           image_array=[image, image], load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        verified = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                           track_id='verified', pin_id='pinpanion:7',
                           review_state='confirmed', identity_confirmed=True)
        saved = BoxData(verified.coords, ['Mario'], target_source,
                        track_id='presence-only', pin_id='pinpanion:7',
                        review_state='inherited', identity_confirmed=False)
        window._ScrubberFrame__frame_boxes.update({0: [verified], 1: [saved]})
        window._ScrubberFrame__detected_frames.add(0)
        window.display_image(reconcile=False)
        with patch('frame_processing.detect_pin_boxes', return_value=[]):
            window.on_play(None)
            window._on_play_timer(None)
            deadline = monotonic() + 5
            while window.current_index != 1 and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert window.current_index == 1
        assert window._ScrubberFrame__frame_boxes[1] == [saved]
        assert saved.track_id == verified.track_id
        assert saved.pin_id == verified.pin_id
        assert saved.review_state == 'inherited' and saved.identity_confirmed
        assert saved.feedback_id is None and saved.feedback_label is None
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
@pytest.mark.parametrize('source_index', [0, 1])
def test_navigation_upgrades_existing_inherited_identity_in_both_directions(source_index):
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(1277).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Adjacent inherited identity', '',
                           image_array=[image, image], load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        target_index = 1 - source_index
        verified = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                           track_id='verified', pin_id='pinpanion:7',
                           review_state='confirmed', identity_confirmed=True)
        saved = BoxData(verified.coords, ['Mario'], 'automatic',
                        track_id='presence-only', pin_id='pinpanion:7',
                        review_state='inherited', identity_confirmed=False)
        window._ScrubberFrame__frame_boxes.update({source_index: [verified],
                                                    target_index: [saved]})
        window._ScrubberFrame__detected_frames.update({0, 1})
        window._current_index = source_index
        window.display_image(reconcile=False)
        if source_index == 0:
            window.on_next_async(None)
        else:
            window.on_prev(None)
        wait_for_reconciliation(window)
        assert saved.track_id == verified.track_id
        assert saved.review_state == 'inherited' and saved.identity_confirmed
        assert saved.pin_id == verified.pin_id
        assert saved.feedback_id is None
        editor = window._ScrubberFrame__tag_panel.find_panel_for_box(saved)
        assert editor.box is saved
        assert editor.confirm_identity_button.GetLabel() == 'Verify here (optional)'
    finally:
        window.Destroy()
        wx.Yield()


def test_rejected_object_tracks_without_reappearing_as_a_candidate():
    old = np.random.default_rng(15).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    current = old.copy()
    rejected = BoxData((10, 15, 40, 40), [], 'automatic', review_state='rejected')
    proposals = [(10, 15, 40, 40), (110, 20, 35, 35)]

    boxes = associate_frame_boxes(old, current, [rejected], proposals)
    assert boxes[0].review_state == 'rejected'
    assert boxes[0].track_id == rejected.track_id
    assert boxes[1].review_state == 'unconfirmed'
    calls = []
    matcher = type('Matcher', (), {'match': lambda self, crop: calls.append(crop.shape)})()
    identify_candidates(current, boxes, matcher)
    assert calls == [(35, 35, 3)]
    assert add_new_detections([rejected], proposals[:1]) == [rejected]
    assert associate_frame_boxes(old, current, [rejected], [])[0].review_state == 'rejected'

    manual_correction = BoxData(rejected.coords, ['Pin'], 'user')
    corrected = associate_frame_boxes(old, current, [rejected, manual_correction],
                                      proposals[:1])[0]
    assert corrected.track_id == manual_correction.track_id
    assert corrected.review_state == 'unconfirmed'

    changed = current.copy()
    changed[15:55, 10:50] = np.random.default_rng(16).integers(0, 255, (40, 40, 3), dtype=np.uint8)
    replacement = associate_frame_boxes(old, changed, [rejected], proposals[:1])[0]
    assert replacement.review_state == 'unconfirmed'


@pytest.mark.parametrize('navigation_index', [0, 2, 4])
def test_rejection_reconciles_separate_saved_tracks_both_directions(navigation_index):
    image = np.random.default_rng(128).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    frames = {index: np.roll(image, index, axis=1) for index in range(5)}
    boxes = {index: [BoxData((10 + index, 15, 40, 40), ['Card design'], 'automatic',
                             track_id=f'candidate-{index}', pin_id='pinpanion:99')]
             for index in range(5)}
    rejected = boxes[2][0]
    rejected.review_state = 'rejected'
    rejected.feedback_label = 'negative'
    unrelated = BoxData((110, 15, 40, 40), ['Mario'], 'automatic',
                        track_id='real-pin', review_state='confirmed')
    boxes[4].append(unrelated)

    assert reconcile_saved_frames(navigation_index, 5, boxes, frames.get)
    assert all(boxes[index][0].review_state == 'rejected' for index in range(5))
    assert len({boxes[index][0].track_id for index in range(5)}) == 1
    assert all(boxes[index][0].tags == ['Card design'] and
               boxes[index][0].pin_id == 'pinpanion:99'
               for index in range(5))
    assert all(boxes[index][0].feedback_label is None for index in (0, 1, 3, 4))
    assert unrelated.review_state == 'confirmed' and unrelated.tags == ['Mario']


def test_rejection_does_not_cross_appearance_change_or_protected_track():
    image = np.random.default_rng(129).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    changed = image.copy()
    changed[15:55, 10:50] = np.random.default_rng(130).integers(
        0, 255, (40, 40, 3), dtype=np.uint8)
    rejected = BoxData((10, 15, 40, 40), [], 'automatic',
                       track_id='rejected', review_state='rejected')
    different = BoxData(rejected.coords, ['Mario'], 'automatic', track_id='different')
    confirmed = BoxData(rejected.coords, ['Mario'], 'automatic',
                        track_id='protected', review_state='confirmed')
    provisional = BoxData(rejected.coords, ['Mario'], 'automatic', track_id='protected')
    frames = {0: [rejected], 1: [different], 2: [provisional], 3: [confirmed]}
    reconcile_saved_frames(0, 4, frames, lambda index: image if index != 1 else changed)
    assert different.review_state == 'unconfirmed'
    assert provisional.review_state == 'inherited'
    assert confirmed.review_state == 'confirmed'


def test_rejection_does_not_suppress_manual_or_confirmed_track():
    image = np.random.default_rng(133).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    rejected = BoxData((10, 15, 40, 40), [], 'automatic',
                       track_id='rejected', review_state='rejected')
    manual = BoxData(rejected.coords, ['Mario'], 'user', track_id='manual')
    provisional = BoxData(rejected.coords, ['Mario'], 'automatic', track_id='confirmed-track')
    confirmed = BoxData(rejected.coords, ['Mario'], 'automatic',
                        track_id='confirmed-track', review_state='confirmed')
    frames = {0: [rejected], 1: [manual, provisional], 2: [confirmed]}
    assert reconcile_saved_frames(0, 3, frames, lambda _index: image)
    assert manual.review_state == 'unconfirmed'
    assert provisional.review_state == 'inherited'
    assert confirmed.review_state == 'confirmed'
    assert provisional.track_id == confirmed.track_id


@pytest.mark.parametrize('source_state,expected_state', [
    ('confirmed', 'inherited'), ('unconfirmed', 'unconfirmed')])
def test_adjacent_identified_pin_fills_previously_checked_empty_frame(
        source_state, expected_state):
    image = np.random.default_rng(134).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     track_id='source', pin_id='pinpanion:7',
                     review_state=source_state)
    frames = {0: [], 1: [source]}
    assert reconcile_saved_frames(0, 2, frames, lambda _index: image,
                                  reference_index=1)
    assert len(frames[0]) == 1
    tracked = frames[0][0]
    assert tracked.coords == source.coords
    assert tracked.track_id == source.track_id
    assert tracked.pin_id == source.pin_id and tracked.tags == source.tags
    assert tracked.review_state == expected_state
    assert tracked.identity_confirmed == (source_state == 'confirmed')


def test_adjacent_identified_pin_does_not_create_box_without_visual_match():
    image = np.random.default_rng(135).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    changed = image.copy()
    changed[15:55, 10:50] = np.random.default_rng(136).integers(
        0, 255, (40, 40, 3), dtype=np.uint8)
    source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed')
    frames = {0: [], 1: [source]}
    assert reconcile_saved_frames(0, 2, frames,
                                  lambda index: image if index == 1 else changed,
                                  reference_index=1)
    assert frames[0] == []


def test_adjacent_pin_keeps_conflicting_manual_label_and_rejected_area():
    image = np.random.default_rng(138).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                     pin_id='pinpanion:7', review_state='confirmed')
    manual = BoxData(source.coords, ['Luigi'], 'user')
    rejected = BoxData(source.coords, [], 'automatic', review_state='rejected')
    for existing in (manual, rejected):
        frames = {0: [existing], 1: [source]}
        assert reconcile_saved_frames(0, 2, frames, lambda _index: image,
                                      reference_index=1)
        assert frames[0] == [existing]
        assert existing.track_id != source.track_id
    assert manual.tags == ['Luigi'] and manual.review_state == 'unconfirmed'
    assert rejected.review_state == 'rejected'


@pytest.mark.gui
@pytest.mark.parametrize('source_index,target_kind', [
    (0, 'empty'), (1, 'empty'), (0, 'automatic'), (1, 'automatic'),
    (0, 'manual'), (1, 'manual')])
def test_navigation_uses_identified_adjacent_pin_on_processed_frame(
        source_index, target_kind):
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(137).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Adjacent navigation', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        target_index = 1 - source_index
        source = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                         pin_id='pinpanion:7', review_state='confirmed')
        saved = (BoxData(source.coords, [''] if target_kind == 'manual' else [],
                         'user' if target_kind == 'manual' else 'automatic')
                 if target_kind != 'empty' else None)
        window._ScrubberFrame__frame_boxes[source_index] = [source]
        window._ScrubberFrame__frame_boxes[target_index] = [saved] if saved else []
        window._ScrubberFrame__detected_frames.update({0, 1})
        window._invalidate_reconciliation()
        assert window.goto_frame(source_index)
        wait_for_reconciliation(window)

        with patch('scrubberframe.detect_pin_boxes', side_effect=AssertionError(
                'reprocessed a checked frame')):
            if source_index == 0:
                window.on_next(None)
            else:
                window.on_prev(None)
            wait_for_reconciliation(window)
        target_boxes = window._ScrubberFrame__frame_boxes[target_index]
        assert len(target_boxes) == 1
        tracked = target_boxes[0]
        if saved is not None:
            assert tracked is saved
        assert tracked.track_id == source.track_id
        assert tracked.pin_id == 'pinpanion:7' and tracked.tags == ['Mario']
        assert tracked.review_state == 'inherited' and tracked.identity_confirmed
        assert window._ScrubberFrame__tag_panel.find_panel_for_box(tracked) is not None
    finally:
        window.Destroy()
        wx.Yield()


def test_processing_next_saved_frame_suppresses_the_same_rejected_object():
    image = np.random.default_rng(132).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    rejected = BoxData((10, 15, 40, 40), [], 'automatic',
                       track_id='rejected', review_state='rejected',
                       feedback_label='negative')
    saved = BoxData(rejected.coords, ['Card design'], 'automatic',
                    track_id='other-track', pin_id='pinpanion:99')
    job = FrameJob(1, 0, [rejected], [saved], [], image, image, None,
                   type('NoRematch', (), {
                       'match': lambda self, crop: pytest.fail('rematched a rejected object')
                   })())
    with patch('frame_processing.detect_pin_boxes', return_value=[]):
        result = process_frame(job, lambda _index: pytest.fail('decoded a cached frame'))
    assert result.boxes[0] is saved
    assert saved.review_state == 'rejected'
    assert saved.track_id == rejected.track_id
    assert saved.tags == ['Card design'] and saved.pin_id == 'pinpanion:99'
    assert saved.feedback_label is None


@pytest.mark.gui
def test_rejection_hides_existing_track_and_suppresses_next_frame():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(17).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Rejection test', '', image_array=[image] * 3,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        before = BoxData((10, 15, 40, 40), [], 'automatic', track_id='false-object')
        current = BoxData((10, 15, 40, 40), [], 'automatic', track_id='false-object')
        window._ScrubberFrame__frame_boxes.update({0: [before], 1: [current]})
        window._ScrubberFrame__detected_frames.update({0, 1})
        window.goto_frame(1)
        panel = window._ScrubberFrame__image_panel
        menu = panel.make_box_context_menu(current)
        not_pin = next(item for item in menu.GetMenuItems()
                       if item.GetItemLabelText() == 'Not a pin')
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, not_pin.GetId()))
        menu.Destroy()
        assert before.review_state == current.review_state == 'rejected'
        assert window._ScrubberFrame__tag_panel.boxes == []
        assert not window.frame_has_boxes(1)

        with patch('scrubberframe.detect_pin_boxes', return_value=[
                (10, 15, 40, 40), (110, 20, 35, 35)]):
            window.on_next(None)
        boxes = window._ScrubberFrame__frame_boxes[2]
        assert [box.review_state for box in boxes] == ['rejected', 'unconfirmed']
        assert len(window._ScrubberFrame__tag_panel.boxes) == 1

        output = StringIO()
        save_boxes_to_stream(output, window._ScrubberFrame__frame_boxes)
        loaded = load_boxes_from_stream(StringIO(output.getvalue()))
        assert [loaded[index][0].review_state for index in (0, 1, 2)] == [
            'rejected', 'rejected', 'rejected']
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_removing_saved_candidate_rejects_separate_tracks_and_undo_restores_them():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(131).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Saved rejection chain', '', image_array=[image] * 5,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        boxes = {index: BoxData((10, 15, 40, 40), ['Card design'], 'automatic',
                                track_id=f'separate-{index}', pin_id='pinpanion:99')
                 for index in range(5)}
        window.goto_frame(2)
        wait_for_reconciliation(window)
        window._ScrubberFrame__frame_boxes.update(
            {index: [box] for index, box in boxes.items()})
        window._ScrubberFrame__detected_frames.update(range(5))
        window._ScrubberFrame__image_panel.boxes = [boxes[2]]
        window._ScrubberFrame__tag_panel.boxes = [boxes[2]]
        window.mark_not_pin(boxes[2])
        wait_for_reconciliation(window)
        assert all(box.review_state == 'rejected' for box in boxes.values())
        assert all(box.tags == ['Card design'] and box.pin_id == 'pinpanion:99'
                   for box in boxes.values())
        assert window._ScrubberFrame__tag_panel.boxes == []

        window.undo_action()
        assert all(box.review_state == 'unconfirmed' for box in boxes.values())
        assert all(box.tags == ['Card design'] and box.pin_id == 'pinpanion:99'
                   for box in boxes.values())
        window.redo_action()
        assert all(box.review_state == 'rejected' for box in boxes.values())
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_detecting_earlier_frame_uses_next_frames_rejection():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(18).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Backward rejection test', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        rejected = BoxData((10, 15, 40, 40), [], 'automatic',
                           review_state='rejected', track_id='false-object')
        window._ScrubberFrame__frame_boxes[1] = [rejected]
        window._ScrubberFrame__detected_frames.add(1)
        with patch('scrubberframe.detect_pin_boxes', return_value=[
                (10, 15, 40, 40), (110, 20, 35, 35)]):
            window._ScrubberFrame__on_process(None)
        boxes = window._ScrubberFrame__frame_boxes[0]
        assert [box.review_state for box in boxes] == ['rejected', 'unconfirmed']
        assert boxes[0].track_id == rejected.track_id
        assert len(window._ScrubberFrame__tag_panel.boxes) == 1
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_label_actions_update_frame_and_saved_boxes():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Review test', '', image_array=[image], load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        box = BoxData((15, 20, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes[0] = [box]
        window.display_image()
        label = window._ScrubberFrame__tag_panel.find_panel_for_box(box)
        assert label is not None
        row = label._BoxTagPanelEdit__box_tags[0]
        assert row.tag == ''
        row._BoxTagLabelRow__text_entry.ChangeValue('Mario')
        row.fire_edited_event()
        wx.Yield()
        assert box.tags == ['Mario']
        label.confirm_button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId,
                                                            label.confirm_button.GetId()))
        assert box.review_state == 'confirmed'
        assert not box.identity_confirmed
        assert ImagePanel.get_box_colour(box) == wx.Colour(0, 160, 60)
        label.delete_button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId,
                                                           label.delete_button.GetId()))
        assert window._ScrubberFrame__frame_boxes[0] == []
        assert window._ScrubberFrame__image_panel.boxes == []
        assert window._ScrubberFrame__tag_panel.boxes == []
        output = StringIO()
        save_boxes_to_stream(output, window._ScrubberFrame__frame_boxes)
        assert load_boxes_from_stream(StringIO(output.getvalue()))[0] == []
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_context_menu_actions_and_playback_pause_on_image_click():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(2).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Playback test', '', image_array=[image] * 3,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        panel = window._ScrubberFrame__image_panel
        from time import monotonic, sleep

        def wait_for_frame(index):
            deadline = monotonic() + 3
            while index not in window._ScrubberFrame__detected_frames and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            assert index in window._ScrubberFrame__detected_frames

        with patch('frame_processing.detect_pin_boxes', return_value=[(15, 20, 40, 40)]):
            window.on_play(None)
            assert window._play_timer.IsRunning()
            window._on_play_timer(None)
            wait_for_frame(0)
            window._on_play_timer(None)
            wait_for_frame(1)
            assert window.current_index == 1
            box = window._ScrubberFrame__frame_boxes[1][0]
            assert ImagePanel.get_box_colour(box) == wx.BLUE
            assert window._ScrubberFrame__tag_panel.boxes == []

            class Click:
                def GetPosition(self):
                    x, y, w, h = panel._box_panel_rect(box)
                    return wx.Point(x + w // 2, y + h // 2)

                def Skip(self):
                    pass

            panel.on_right_down(Click())
            assert not window._play_timer.IsRunning()
            assert window._ScrubberFrame__tag_panel.boxes == [box]
            assert window._ScrubberFrame__button_panel.play_btn.GetLabel() == 'Play'
            menu = panel.make_box_context_menu(box)
            assert [item.GetItemLabelText() for item in menu.GetMenuItems()] == [
                'Confirm is a pin', 'Confirm pin identity', 'Remove area', 'Not a pin']
            item = menu.GetMenuItems()[0]
            menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item.GetId()))
            menu.Destroy()
            assert box.review_state == 'confirmed'

            window.on_play(None)
            window._on_play_timer(None)
            wait_for_frame(2)
            assert window.current_index == 2
            assert not window._play_timer.IsRunning()
            assert window._ScrubberFrame__frame_boxes[2][0].review_state == 'inherited'
            assert ImagePanel.get_box_colour(window._ScrubberFrame__frame_boxes[2][0]) == wx.Colour(230, 140, 0)
            assert window._ScrubberFrame__detected_frames == {0, 1, 2}

            last = window._ScrubberFrame__frame_boxes[2][0]
            menu = panel.make_box_context_menu(last)
            item = menu.GetMenuItems()[2]
            menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, item.GetId()))
            menu.Destroy()
            assert window._ScrubberFrame__frame_boxes[2] == []
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_manual_box_has_blank_edit_field_and_separate_identity_confirmation():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Manual pin review', '', image_array=[image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        window.SetSize((800, 600))
        window.Layout()
        window.display_image()
        image_panel = window._ScrubberFrame__image_panel
        tag_panel = window._ScrubberFrame__tag_panel
        image_panel.add_new_box((15, 20, 40, 40))
        wx.Yield()
        box = window._ScrubberFrame__frame_boxes[0][0]
        assert box.tags == [''] and box.pin_id is None
        editor = tag_panel.find_panel_for_box(box)
        assert editor is not None
        row = editor._BoxTagPanelEdit__box_tags[0]
        assert row.tag == ''
        assert editor.IsShown()

        menu = image_panel.make_box_context_menu(box)
        presence, identity, _remove = menu.GetMenuItems()
        assert not presence.IsEnabled() and not identity.IsEnabled()
        menu.Destroy()
        assert box.review_state == 'confirmed'
        assert box.feedback_label == 'positive' and box.feedback_id
        assert not box.identity_confirmed
        assert box.tags == ['']
        saved = StringIO()
        save_boxes_to_stream(saved, window._ScrubberFrame__frame_boxes)
        assert not load_boxes_from_stream(StringIO(saved.getvalue()))[0][0].identity_confirmed

        row._BoxTagLabelRow__text_entry.ChangeValue('Mario')
        row.fire_edited_event()
        wx.Yield()
        assert box.tags == ['Mario']
        assert not box.identity_confirmed
        menu = image_panel.make_box_context_menu(box)
        presence, identity, _wrong, _remove = menu.GetMenuItems()
        assert not presence.IsEnabled() and identity.IsEnabled()
        menu.ProcessEvent(wx.CommandEvent(wx.EVT_MENU.typeId, identity.GetId()))
        menu.Destroy()
        assert box.identity_confirmed
        assert box.tags == ['Mario']
        assert tag_panel.find_panel_for_box(box) is editor
        inherited = BoxData(box.coords, ['Mario'], 'automatic',
                            review_state='inherited', identity_confirmed=True)
        inherited_menu = image_panel.make_box_context_menu(inherited)
        assert inherited_menu.GetMenuItems()[1].IsEnabled()
        inherited_menu.Destroy()
        saved = StringIO()
        save_boxes_to_stream(saved, window._ScrubberFrame__frame_boxes)
        assert load_boxes_from_stream(StringIO(saved.getvalue()))[0][0].identity_confirmed
        window.undo_action()
        assert box.review_state == 'confirmed' and not box.identity_confirmed
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_background_detection_does_not_block_ui_and_pause_discards_result():
    from threading import Event, get_ident
    from time import monotonic

    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(81).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Async playback', '', image_array=[image] * 2,
                           load_catalog=False)
    started = Event()
    release = Event()
    worker_thread = []

    def slow_detect(*_args, **_kwargs):
        worker_thread.append(get_ident())
        started.set()
        assert release.wait(3)
        return [(15, 20, 40, 40)]

    try:
        window.Show()
        wx.Yield()
        ui_thread = get_ident()
        with patch('frame_processing.detect_pin_boxes', side_effect=slow_detect):
            window.on_play(None)
            before = monotonic()
            window._on_play_timer(None)
            assert monotonic() - before < 0.2
            assert started.wait(2)
            assert len(worker_thread) == 1
            assert worker_thread[0] != ui_thread
            future = window._pending_frame_job
            pause_button = window._ScrubberFrame__button_panel.play_btn
            click = wx.CommandEvent(wx.EVT_BUTTON.typeId, pause_button.GetId())
            click.SetEventObject(pause_button)
            before = monotonic()
            pause_button.ProcessEvent(click)
            assert monotonic() - before < 0.2
            assert pause_button.GetLabel() == 'Play'
            assert not window._playing
            release.set()
            future.result(timeout=2)
            wx.Yield()
            assert window._ScrubberFrame__detected_frames == set()
            assert window._ScrubberFrame__frame_boxes.get(0, []) == []
    finally:
        release.set()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_next_and_previous_scrub_immediately_while_detection_is_busy():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(139).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Responsive navigation', '', image_array=[image] * 3,
                           load_catalog=False)
    started = Event()
    release = Event()

    def slow_detect(*_args, **_kwargs):
        started.set()
        assert release.wait(5)
        return [(10, 15, 40, 40)]

    try:
        window.Show()
        wx.Yield()
        controls = window._ScrubberFrame__button_panel

        def click(button):
            event = wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId())
            event.SetEventObject(button)
            button.ProcessEvent(event)

        with patch('frame_processing.detect_pin_boxes', side_effect=slow_detect):
            click(controls.next_btn)
            assert window.current_index == 1
            assert window.slider.GetValue() == 1
            first_future = window._pending_frame_job
            assert started.wait(2)
            click(controls.next_btn)
            assert window.current_index == 2
            assert window.slider.GetValue() == 2
            click(controls.prev_btn)
            assert window.current_index == 1
            assert window.slider.GetValue() == 1
            release.set()
            first_future.result(timeout=5)
            deadline = monotonic() + 5
            while window._pending_frame_job is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
            assert window.current_index == 1
    finally:
        release.set()
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_navigation_reports_decode_failure_and_keeps_frame_controls_correct():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(140).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Decode failure navigation', '', image_array=[image] * 2,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        window._ScrubberFrame__detected_frames.add(1)
        real_get_frame = window.get_frame
        with patch.object(window, 'get_frame', side_effect=lambda index, rotation=0:
                          None if index == 1 else real_get_frame(index, rotation)):
            window.on_next_async(None)
            assert window.current_index == 1
            assert window.slider.GetValue() == 1
            assert 'could not decode image' in window.GetStatusBar().GetStatusText(1)
            assert window._ScrubberFrame__button_panel.prev_btn.IsEnabled()
            window.on_prev(None)
        assert window.current_index == 0
        assert window.slider.GetValue() == 0
        assert window._ScrubberFrame__button_panel.next_btn.IsEnabled()
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_playback_reconciles_saved_adjacent_box_without_ui_chain_scan():
    from time import monotonic, sleep

    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(82).integers(0, 255, (120, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Playback chain', '', image_array=[image, image],
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        source = BoxData((25, 25, 40, 40), ['Mario'], 'automatic',
                         review_state='confirmed')
        saved = BoxData((25, 25, 40, 40), [], 'automatic')
        window._ScrubberFrame__frame_boxes.update({0: [source], 1: [saved]})
        window._ScrubberFrame__detected_frames.add(0)
        with patch('frame_processing.detect_pin_boxes', return_value=[]), \
             patch.object(window, '_reconcile_frame',
                          side_effect=AssertionError('chain scan ran on UI')):
            window.on_play(None)
            window._on_play_timer(None)
            deadline = monotonic() + 3
            while window.current_index != 1 and monotonic() < deadline:
                wx.Yield()
                sleep(0.005)
        assert window.current_index == 1
        assert window._ScrubberFrame__frame_boxes[1][0] is saved
        assert saved.track_id == source.track_id
        assert saved.tags == ['Mario']
        assert saved.review_state == 'inherited'
        assert not window._playing
    finally:
        window.Destroy()
        wx.Yield()


@pytest.mark.gui
def test_long_saved_chain_decodes_on_worker_and_resize_does_not_restart_it():
    app = wx.App.Get() or wx.App(False)
    image = np.random.default_rng(83).integers(0, 255, (100, 180, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Long chain', '', image_array=[image] * 92,
                           load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        boxes = window._ScrubberFrame__frame_boxes
        for index in range(39, 91):
            boxes[index] = [BoxData((10, 15, 40, 40), [], 'automatic')]
        known = BoxData((10, 15, 40, 40), ['Mario'], 'automatic',
                        review_state='confirmed')
        boxes[91] = [known]
        first_read = Event()
        allow_read = Event()
        real_worker_read = window.read_frame_for_processing

        def delayed_worker_read(index):
            first_read.set()
            assert allow_read.wait(3)
            return real_worker_read(index)

        with patch.object(window, 'get_frame', wraps=window.get_frame) as ui_read, \
             patch.object(window, 'read_frame_for_processing', side_effect=delayed_worker_read):
            assert window.goto_frame(39)
            assert first_read.wait(2)
            pending = window._pending_reconcile

            class Resize:
                def Skip(self):
                    pass

            window.on_resize(Resize())
            assert window._pending_reconcile is pending
            assert [call.args[0] for call in ui_read.call_args_list] == [39]
            allow_read.set()
            wait_for_reconciliation(window)
        assert boxes[39][0].track_id == known.track_id
        assert boxes[39][0].review_state == 'inherited'
        assert boxes[39][0].tags == ['Mario']
    finally:
        allow_read.set()
        window.Destroy()
        wx.Yield()
