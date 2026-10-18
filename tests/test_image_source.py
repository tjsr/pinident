import cv2
import numpy as np
import pytest
import wx

from videoscrubber import VideoScrubber


@pytest.mark.gui
def test_editor_can_review_still_image_with_persistent_source(tmp_path):
    app = wx.App.Get() or wx.App(False)
    rgb = np.random.default_rng(10).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    image_path = tmp_path / 'pin-photo.png'
    assert cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    window = VideoScrubber(None, 'Image review', '', video_path=str(image_path),
                           load_catalog=False,
                           feedback_path=tmp_path / 'feedback.sqlite3')
    try:
        assert window._ScrubberFrame__num_frames == 1
        assert window._VideoScrubber__cap is None
        assert window.box_data_filename == str(tmp_path / 'pin-photo.json')
        assert window._feedback_source == str(image_path.resolve())
        assert np.array_equal(window.get_frame(0), rgb)
        assert np.array_equal(window.read_frame_for_processing(0), rgb)
    finally:
        window.Destroy()
        wx.Yield()
