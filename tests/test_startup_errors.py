from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError
from time import monotonic, sleep

import numpy as np
import pytest
import wx

from cloud_sync import CloudConfig, CloudSyncError, fetch_model
from scrubberframe import ScrubberFrame
from videoscrubber import VideoScrubber


def test_resize_before_slider_exists_skips_display():
    source = np.zeros((8, 8, 3), dtype=np.uint8)
    frame = SimpleNamespace(_display_source_frame=source, display_image=Mock())
    event = SimpleNamespace(Skip=Mock())

    ScrubberFrame.on_resize(frame, event)

    frame.display_image.assert_not_called()
    event.Skip.assert_called_once_with()


def test_unavailable_startup_cloud_sync_is_reported_without_traceback():
    future = Future()
    future.set_exception(CloudSyncError('Cloudflare sync is unavailable: DNS lookup failed'))
    frame = SimpleNamespace(
        _closing=False, _cloud_sync_future=future, _cloud_train_after_sync=False,
        _cloud_ready_for_publish=True, _cloud_status='', SetStatusText=Mock())
    logger = Mock()

    with patch('scrubberframe.getLog', return_value=logger):
        ScrubberFrame._finish_cloud_sync(frame, future)

    assert frame._cloud_sync_future is None
    assert not frame._cloud_ready_for_publish
    assert 'Unavailable' in frame._cloud_status
    logger.warning.assert_called_once()
    logger.exception.assert_not_called()


def test_dns_error_is_classified_as_unavailable_cloud_sync():
    with patch('cloud_sync.urlopen', side_effect=URLError('DNS lookup failed')):
        with pytest.raises(CloudSyncError, match='DNS lookup failed'):
            fetch_model(CloudConfig('https://id.pinpanion.com', None))


@pytest.mark.gui
def test_window_opens_with_early_resize_and_unavailable_cloud(tmp_path, capsys):
    app = wx.App.Get() or wx.App(False)
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    with patch('scrubberframe.fetch_model',
               side_effect=CloudSyncError('Cloudflare sync is unavailable: DNS lookup failed')):
        window = VideoScrubber(
            None, 'Startup', '', image_array=[image], load_catalog=False,
            catalog_path=str(tmp_path / 'catalog.sqlite3'),
            feedback_path=tmp_path / 'feedback.sqlite3')
        try:
            window.Show()
            deadline = monotonic() + 3
            while window._cloud_sync_future is not None and monotonic() < deadline:
                wx.Yield()
                sleep(0.01)
            wx.Yield()
            assert window.slider is not None
            assert app is not None
            assert window._cloud_sync_future is None
            assert window._cloud_status.startswith('Unavailable')
        finally:
            window.Destroy()
            wx.Yield()
    assert 'AttributeError' not in capsys.readouterr().err
