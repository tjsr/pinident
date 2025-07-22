import cv2
import wx

from scrubberframe import ScrubberFrame

class VideoScrubber(ScrubberFrame):
    @property
    def current_index(self):
        return self._current_index

    def __init__(self, parent, title, video_path=None, image_array=None, box_data: str | None = None):
        self.cap = None
        self.num_frames = 0
        self.image_array = image_array
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        elif image_array:
            self.num_frames = len(image_array)
        else:
            raise ValueError("Either video_path or image_array must be provided.")
        super().__init__(parent, title, self.num_frames)
        self.get_frame(0)

        if video_path is not None:
            self.box_data_filename = self.create_box_data_name_from_filename(video_path)


    # @ScrubberFrame.current_index.setter
    # def current_index(self, index):
    #     super(VideoScrubber, type(self))._current_index.__set__(self, index)
    #     self.get_frame(self._current_index)
    #     self.display_image()

    def get_frame(self, index, rotation_angle: int = 0):
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ret, frame = self.cap.read()
            if not ret:
                return None
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif self.image_array:
            img = cv2.cvtColor(self.image_array[index], cv2.COLOR_BGR2RGB)
        else:
            return None

        if rotation_angle % 360 == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle % 360 == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation_angle % 360 == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        self.Layout()
        self.Refresh()

        return img

    def __del__(self):
        if self.cap:
            self.cap.release()

