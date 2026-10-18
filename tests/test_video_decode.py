from unittest.mock import patch
from time import monotonic, sleep

import numpy as np
import pytest
import wx

from videoscrubber import VideoScrubber
from frame_cache import FrameCache


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

        # Keep the decoder-specific sequential-read assertion independent of
        # the shared UI/worker frame cache.
        window._frame_cache = FrameCache()
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


def test_frame_cache_evicts_least_recently_used_frames_by_byte_budget():
    cache = FrameCache(max_bytes=12)
    frames = [np.full((2, 2, 1), index, dtype=np.uint8) for index in range(4)]
    for index in range(3):
        cache.put(index, frames[index])
    assert cache.used_bytes == 12
    assert cache.get(0) is frames[0]
    cache.put(3, frames[3])
    assert cache.get(1) is None
    assert cache.get(0) is frames[0]
    assert cache.used_bytes == 12


@pytest.mark.gui
def test_prefetched_frame_is_shared_with_ui_and_processing_worker():
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    window = VideoScrubber(None, 'Prefetch test', '', image_array=[image] * 6,
                           load_catalog=False)
    ui_capture = FakeCapture()
    prefetch_capture = FakeCapture()
    try:
        window._VideoScrubber__cap = ui_capture
        window._video_path = 'fake.avi'
        with patch('videoscrubber.cv2.VideoCapture', return_value=prefetch_capture):
            assert window.get_frame(0)[0, 0, 0] == 0
            deadline = monotonic() + 2
            while window._frame_cache.get(2) is None and monotonic() < deadline:
                sleep(0.01)
            assert window._frame_cache.get(2) is not None
            assert window.get_frame(2)[0, 0, 0] == 2
            assert window.read_frame_for_processing(2)[0, 0, 0] == 2
        assert ui_capture.reads == 1
        assert prefetch_capture.reads >= 2
    finally:
        window._stop_prefetch()
        window.Destroy()
        wx.Yield()
