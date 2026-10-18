"""A byte-bounded cache for decoded, unrotated RGB video frames."""

from collections import OrderedDict
from threading import Lock

import numpy as np


DEFAULT_FRAME_CACHE_BYTES = 128 * 1024 * 1024


class FrameCache:
    def __init__(self, max_bytes: int = DEFAULT_FRAME_CACHE_BYTES):
        if max_bytes < 0:
            raise ValueError('Frame cache size must be non-negative')
        self.max_bytes = max_bytes
        self._frames: OrderedDict[int, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = Lock()

    def get(self, index: int) -> np.ndarray | None:
        with self._lock:
            frame = self._frames.get(index)
            if frame is not None:
                self._frames.move_to_end(index)
            return frame

    def put(self, index: int, frame: np.ndarray) -> None:
        if frame.nbytes > self.max_bytes:
            return
        with self._lock:
            old = self._frames.pop(index, None)
            if old is not None:
                self._bytes -= old.nbytes
            self._frames[index] = frame
            self._bytes += frame.nbytes
            while self._bytes > self.max_bytes:
                _, old = self._frames.popitem(last=False)
                self._bytes -= old.nbytes

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._bytes
