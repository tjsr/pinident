"""Bulk confirmation of catalog suggestions in the current video frame."""

from io import StringIO

import cv2
import numpy as np
import pytest
import wx

from boxdata import BoxData
from scrubberframe import save_boxes_to_stream
from videoscrubber import VideoScrubber


@pytest.mark.gui
def test_confirm_all_suggestions_only_verifies_visible_identified_areas():
    app = wx.App.Get() or wx.App(False)
    rgb = np.random.default_rng(721).integers(0, 256, (160, 320, 3), dtype=np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    window = VideoScrubber(None, 'Confirm suggested pins', '',
                           image_array=[bgr, bgr.copy()], load_catalog=False)
    try:
        window.Show()
        wx.Yield()
        suggested = BoxData((5, 5, 30, 30), ['Mario'], 'automatic',
                            pin_id='pinpanion:1', match_confidence=0.8)
        blue = BoxData((45, 5, 30, 30), [], 'automatic')
        unnamed = BoxData((85, 5, 30, 30), [''], 'user',
                          review_state='confirmed', identity_confirmed=False)
        presence = BoxData((125, 5, 30, 30), ['Luigi'], 'automatic',
                           pin_id='pinpanion:2', review_state='confirmed',
                           identity_confirmed=False)
        inherited = BoxData((165, 5, 30, 30), ['Peach'], 'automatic',
                            pin_id='pinpanion:3', review_state='inherited',
                            identity_confirmed=False)
        typed = BoxData((205, 5, 30, 30), ['Typed name'], 'automatic',
                        identity_confirmed=False)
        rejected = BoxData((245, 5, 30, 30), ['Wrong'], 'automatic',
                           pin_id='pinpanion:4', review_state='rejected')
        verified = BoxData((5, 55, 30, 30), ['Verified'], 'automatic',
                           pin_id='pinpanion:5', review_state='confirmed',
                           identity_confirmed=True)
        placeholder = BoxData((85, 55, 30, 30), ['unknown-1'], 'automatic',
                              pin_id='unknown-1', identity_confirmed=False)
        next_frame = BoxData((45, 55, 30, 30), ['Other'], 'automatic',
                             pin_id='pinpanion:6')
        current = [suggested, blue, unnamed, presence, inherited, typed,
                   rejected, verified, placeholder]
        window._ScrubberFrame__frame_boxes[0] = current
        window._ScrubberFrame__frame_boxes[1] = [next_frame]
        window.display_image(reconcile=False)

        button = window._ScrubberFrame__button_panel.confirm_all_btn
        assert button.GetLabel() == 'Confirm all suggestions'
        assert button.IsShownOnScreen()
        assert button.GetScreenRect().Bottom <= window.GetScreenRect().Bottom
        button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId()))

        for box in (suggested, presence, inherited):
            assert box.review_state == 'confirmed'
            assert box.identity_confirmed
            assert box.feedback_label == 'positive'
        assert blue.review_state == 'unconfirmed' and not blue.identity_confirmed
        assert unnamed.review_state == 'confirmed' and not unnamed.identity_confirmed
        assert typed.review_state == 'unconfirmed' and not typed.identity_confirmed
        assert rejected.review_state == 'rejected' and not rejected.identity_confirmed
        assert placeholder.review_state == 'unconfirmed' and not placeholder.identity_confirmed
        assert verified.review_state == 'confirmed' and verified.identity_confirmed
        assert next_frame.review_state == 'unconfirmed' and not next_frame.identity_confirmed
        assert len(window._undo_history[0]) == 1
        assert window._feedback_store.counts()['positive'] == 3
        button.ProcessEvent(wx.CommandEvent(wx.EVT_BUTTON.typeId, button.GetId()))
        assert len(window._undo_history[0]) == 1
        saved = StringIO()
        save_boxes_to_stream(saved, window._ScrubberFrame__frame_boxes)
        assert saved.getvalue().count('"identity_confirmed": true') == 4

        window.undo_action()
        assert all(not box.identity_confirmed for box in (suggested, presence, inherited))
        assert [box.review_state for box in (suggested, presence, inherited)] == [
            'unconfirmed', 'confirmed', 'inherited']
        assert window._feedback_store.counts()['positive'] == 0
        window.redo_action()
        assert all(box.identity_confirmed for box in (suggested, presence, inherited))
        assert window._feedback_store.counts()['positive'] == 3
    finally:
        window.Destroy()
        wx.Yield()
