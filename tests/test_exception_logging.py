from io import StringIO
import logging
import sys
import threading
from pathlib import Path

import pytest
import wx

import logutil


def test_console_logger_uses_live_stream_after_capture_closes(
        monkeypatch: pytest.MonkeyPatch) -> None:
    closed = StringIO()
    closed.close()
    fallback = StringIO()
    monkeypatch.setattr(sys, 'stdout', closed)
    monkeypatch.setattr(sys, '__stderr__', fallback)
    handler = logutil.DynamicStreamHandler()
    handler.emit(logging.LogRecord('Pinident', logging.ERROR, __file__, 1,
                                   'callback failure', (), None))
    assert 'callback failure' in fallback.getvalue()


@pytest.mark.gui
def test_wx_callback_exception_writes_traceback_to_diagnostic_log(
        tmp_path: Path) -> None:
    original_exception_hook = sys.excepthook
    original_thread_hook = threading.excepthook
    app = wx.App.Get() or wx.App(False)
    frame = wx.Frame(None, title='Exception diagnostics')
    log_path = tmp_path / 'pinident.log'
    try:
        logutil.install_exception_logging(log_path)
        frame.Show()

        def raise_from_callback() -> None:
            raise RuntimeError('review callback failed')

        wx.CallAfter(raise_from_callback)
        app.Yield()

        contents = log_path.read_text(encoding='utf-8')
        assert 'Unhandled application exception' in contents
        assert 'Traceback (most recent call last):' in contents
        assert 'raise_from_callback' in contents
        assert 'RuntimeError: review callback failed' in contents

        def raise_from_thread() -> None:
            raise ValueError('worker failed')

        worker = threading.Thread(target=raise_from_thread, name='diagnostic-worker')
        worker.start()
        worker.join()
        contents = log_path.read_text(encoding='utf-8')
        assert 'Unhandled exception in thread diagnostic-worker' in contents
        assert 'ValueError: worker failed' in contents
    finally:
        frame.Destroy()
        sys.excepthook = original_exception_hook
        threading.excepthook = original_thread_hook
        handler = logutil._exception_file_handler
        if handler is not None:
            logging.getLogger().removeHandler(handler)
            handler.close()
            logutil._exception_file_handler = None
