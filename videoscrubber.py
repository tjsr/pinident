import cv2
import wx
from pathlib import Path
from time import perf_counter
from logutil import getLog

from scrubberframe import ScrubberFrame

class VideoScrubber(ScrubberFrame):
    __cap: cv2.VideoCapture | None

    @property
    def current_index(self):
        return self._current_index

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
        self.image_array = image_array
        if video_path:
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
            if index == self._ui_last_index and self._ui_last_frame is not None:
                img = self._ui_last_frame
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
        else:
            frame = self.image_array[index]
            result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        getLog().info('frame=%d decode=%.1fms seek=%.1fms mode=%s thread=worker',
                      index, (perf_counter() - started) * 1000, seek_ms, mode)
        return result

    def close_processing_reader(self) -> None:
        if self._processing_cap is not None:
            self._processing_cap.release()
            self._processing_cap = None
        self._processing_last_index = -1
        self._processing_last_frame = None

    def __del__(self):
        if self.__cap:
            self.__cap.release()
        self.close_processing_reader()
