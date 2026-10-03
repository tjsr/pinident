from unittest.mock import patch

import numpy as np
import pytest
import wx

from videoscrubber import VideoScrubber


class FakeCapture:
    def __init__(self):
        self.position = 0
        self.seeks = []
        self.reads = 0

    def isOpened(self):
        return True

    def set(self, property_id, value):
        self.position = int(value)
        self.seeks.append(self.position)
        return True

    def read(self):
        self.reads += 1
        frame = np.full((32, 32, 3), self.position, dtype=np.uint8)
        self.position += 1
        return True, frame

    def release(self):
        pass


@pytest.mark.gui
def test_video_decoders_cache_current_and_read_adjacent_frames_without_seek():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Decode test', '', image_array=[image] * 4,
                           load_catalog=False)
    try:
        ui_capture = FakeCapture()
        window._VideoScrubber__cap = ui_capture
        assert window.get_frame(0)[0, 0, 0] == 0
        assert window.get_frame(0)[0, 0, 0] == 0
        assert window.get_frame(1)[0, 0, 0] == 1
        assert window.get_frame(3)[0, 0, 0] == 3
        assert ui_capture.seeks == [3]
        assert ui_capture.reads == 3

        worker_capture = FakeCapture()
        window._video_path = 'fake.avi'
        with patch('videoscrubber.cv2.VideoCapture', return_value=worker_capture):
            assert window.read_frame_for_processing(0)[0, 0, 0] == 0
            assert window.read_frame_for_processing(0)[0, 0, 0] == 0
            assert window.read_frame_for_processing(1)[0, 0, 0] == 1
            assert window.read_frame_for_processing(3)[0, 0, 0] == 3
        assert worker_capture.seeks == [3]
        assert worker_capture.reads == 3
    finally:
        window.Destroy()
        wx.Yield()
