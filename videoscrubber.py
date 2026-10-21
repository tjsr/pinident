from math import isfinite
from pathlib import Path
from threading import Condition, Thread, current_thread
from time import perf_counter

import cv2
import wx
from frame_cache import FrameCache
from logutil import getLog

from scrubberframe import ScrubberFrame

class VideoScrubber(ScrubberFrame):
    __cap: cv2.VideoCapture | None
    IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    PREFETCH_FRAMES = 4

    @property
    def current_index(self) -> int:
        return self._current_index

    def playback_frame_interval_ms(self) -> int:
        """Use the source frame rate to pace work that finishes quickly."""
        if self.__cap is None:
            return 50
        fps = float(self.__cap.get(cv2.CAP_PROP_FPS))
        return max(1, round(1000 / fps)) if isfinite(fps) and fps > 0 else 50

    def __init__(self, parent: wx.Panel | None, title: str, opencv_exe_path: str,
                 video_path=None, image_array=None, box_data: str | None = None,
                 catalog_path: str | None = None, load_catalog: bool = True,
                 feedback_path: str | Path | None = None):
        self.__cap = None
        self._ui_last_index = -1
        self._ui_last_frame = None
        self._video_path = video_path
        self._processing_cap = None
        self._processing_last_index = -1
        self._processing_last_frame = None
        self._frame_cache = FrameCache()
        self._prefetch_condition = Condition()
        self._prefetch_target = -1
        self._prefetch_revision = 0
        self._prefetch_stopped = False
        self._prefetch_thread = None
        self.image_array = image_array
        if video_path and Path(video_path).suffix.lower() in self.IMAGE_SUFFIXES:
            still = cv2.imread(str(video_path), cv2.IMREAD_COLOR)
            if still is None:
                raise FileNotFoundError(f'Cannot open image: {video_path}')
            self.image_array = [still]
            self._video_path = None
            num_frames = 1
        elif video_path:
            self.__cap = cv2.VideoCapture(video_path)
            if not self.__cap.isOpened():
                self.__cap.release()
                raise FileNotFoundError(f'Cannot open video: {video_path}')
            num_frames = int(self.__cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif image_array:
            num_frames = len(image_array)
        else:
            raise ValueError("Either video_path or image_array must be provided.")
        if video_path and feedback_path is None:
            from detection_feedback import DEFAULT_FEEDBACK_DB
            feedback_path = DEFAULT_FEEDBACK_DB
        super().__init__(parent, title, opencv_exe_path, num_frames, catalog_path,
                         load_catalog, feedback_path,
                         str(Path(video_path).resolve()) if video_path else None)
        if video_path is not None:
            self.box_data_filename = self.create_box_data_name_from_filename(video_path)

    # @ScrubberFrame.current_index.setter
    # def current_index(self, index):
    #     super(VideoScrubber, type(self))._current_index.__set__(self, index)
    #     self.get_frame(self._current_index)
    #     self.display_image()

    def get_frame(self, index, rotation_angle: int = 0):
        started = perf_counter()
        mode = 'array'
        seek_ms = 0.0
        if self.__cap:
            if not 0 <= index < self._ScrubberFrame__num_frames:
                return None
            img = self._frame_cache.get(index)
            if img is not None:
                mode = 'cache'
            else:
                mode = 'sequential' if index == self._ui_last_index + 1 else 'seek'
                if mode == 'seek':
                    self.__cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    seek_ms = (perf_counter() - started) * 1000
                ret, frame = self.__cap.read()
                if not ret:
                    self._ui_last_index = -1
                    self._ui_last_frame = None
                    return None
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._ui_last_index = index
                self._ui_last_frame = img
                self._frame_cache.put(index, img)
            self._schedule_prefetch(index)
        elif self.image_array:
            if not 0 <= index < len(self.image_array):
                return None
            img = cv2.cvtColor(self.image_array[index], cv2.COLOR_BGR2RGB)
        else:
            return None

        if rotation_angle % 360 == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle % 360 == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation_angle % 360 == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        getLog().info('frame=%d decode=%.1fms seek=%.1fms mode=%s thread=ui',
                      index, (perf_counter() - started) * 1000, seek_ms, mode)
        return img

    def read_frame_for_processing(self, index: int):
        """Decode away from wx using a capture separate from the UI capture."""
        started = perf_counter()
        mode = 'array'
        seek_ms = 0.0
        if not 0 <= index < self._ScrubberFrame__num_frames:
            return None
        if self._video_path:
            cached = self._frame_cache.get(index)
            if cached is not None:
                getLog().info('frame=%d decode=%.1fms seek=0.0ms mode=cache thread=worker',
                              index, (perf_counter() - started) * 1000)
                return cached
            if self._processing_cap is None:
                self._processing_cap = cv2.VideoCapture(self._video_path)
                if not self._processing_cap.isOpened():
                    return None
            if index == self._processing_last_index and self._processing_last_frame is not None:
                result = self._processing_last_frame
                mode = 'cache'
            else:
                mode = ('sequential' if index == self._processing_last_index + 1
                        else 'seek')
                if mode == 'seek':
                    seek_started = perf_counter()
                    self._processing_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    seek_ms = (perf_counter() - seek_started) * 1000
                ok, frame = self._processing_cap.read()
                if not ok:
                    self._processing_last_index = -1
                    self._processing_last_frame = None
                    return None
                result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self._processing_last_index = index
                self._processing_last_frame = result
                self._frame_cache.put(index, result)
        else:
            frame = self.image_array[index]
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        getLog().info('frame=%d decode=%.1fms seek=%.1fms mode=%s thread=worker',
                      index, (perf_counter() - started) * 1000, seek_ms, mode)
        return result

    def _schedule_prefetch(self, index: int) -> None:
        if not self._video_path or index >= self._ScrubberFrame__num_frames - 1:
            return
        with self._prefetch_condition:
            if self._prefetch_stopped or index == self._prefetch_target:
                return
            self._prefetch_target = index
            self._prefetch_revision += 1
            if self._prefetch_thread is None:
                self._prefetch_thread = Thread(
                    target=self._prefetch_frames, name='pin-frame-prefetch', daemon=True)
                self._prefetch_thread.start()
            self._prefetch_condition.notify()

    def _prefetch_frames(self) -> None:
        """Decode a short forward window with a capture owned by this thread."""
        capture = None
        last_index = -1
        seen_revision = 0
        try:
            while True:
                with self._prefetch_condition:
                    self._prefetch_condition.wait_for(
                        lambda: self._prefetch_stopped or
                        self._prefetch_revision != seen_revision)
                    if self._prefetch_stopped:
                        return
                    seen_revision = self._prefetch_revision
                    target = self._prefetch_target
                end = min(target + self.PREFETCH_FRAMES + 1,
                          self._ScrubberFrame__num_frames)
                for index in range(target + 1, end):
                    with self._prefetch_condition:
                        if self._prefetch_stopped or self._prefetch_revision != seen_revision:
                            break
                    if self._frame_cache.get(index) is not None:
                        continue
                    if capture is None:
                        capture = cv2.VideoCapture(self._video_path)
                        if not capture.isOpened():
                            getLog().warning('Could not open video for frame prefetch')
                            return
                    if index != last_index + 1:
                        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, frame = capture.read()
                    if not ok:
                        last_index = -1
                        break
                    last_index = index
                    self._frame_cache.put(index, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception:
            getLog().exception('Background frame prefetch failed')
        finally:
            if capture is not None:
                capture.release()

    def _stop_prefetch(self, timeout: float = 1) -> None:
        with self._prefetch_condition:
            self._prefetch_stopped = True
            self._prefetch_condition.notify_all()
        if self._prefetch_thread is not None and self._prefetch_thread is not current_thread():
            self._prefetch_thread.join(timeout=timeout)

    def _shutdown_stop_prefetch(self) -> bool:
        self._stop_prefetch(timeout=0)
        return self._prefetch_thread is None or not self._prefetch_thread.is_alive()

    def _shutdown_resume_prefetch(self) -> None:
        with self._prefetch_condition:
            if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
                return
            self._prefetch_thread = None
            self._prefetch_stopped = False
            self._prefetch_target = -1
        self._schedule_prefetch(self._current_index)

    def close_processing_reader(self) -> None:
        if self._processing_cap is not None:
            self._processing_cap.release()
            self._processing_cap = None
        self._processing_last_index = -1
        self._processing_last_frame = None

    def __del__(self):
        if hasattr(self, '_prefetch_thread'):
            self._stop_prefetch()
        if self.__cap:
            self.__cap.release()
        self.close_processing_reader()
