import cv2
import os

from scrubberframe import ScrubberFrame

class ImageScrubber(ScrubberFrame):
    def __init__(self, parent, title, image_dir):
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        self.image_files.sort()
        self.image_dir = image_dir
        super().__init__(parent, title, len(self.image_files))

    def get_frame(self, index: int, rotation_angle: int = 0):
        if not self.image_files:
            return None
        img_path = os.path.join(self.image_dir, self.image_files[index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rotation_angle % 360 == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_angle % 360 == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif rotation_angle % 360 == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img