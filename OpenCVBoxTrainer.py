import os
import subprocess
import tempfile
from typing import List

import cv2
import numpy as np

from boxdata import BoxData

OPENCV_BASE_URL = 'https://github.com/opencv/opencv/releases/'
OPENCV_LATEST_VERSION = '4.12'
OPENCV_URL = f'{OPENCV_BASE_URL}download/{OPENCV_LATEST_VERSION}/opencv-{OPENCV_LATEST_VERSION}-windows.exe'
REQUIRES_EXE = False

class OpenCVBoxTrainer:
	__exe_path: str

	def __check_exe_exists(self) -> bool:
		if not os.path.isfile(self.__exe_path):
			print('OpenCV executable not found at:', self.__exe_path)
			print(f'It can be downloaded from {OPENCV_URL}')
			raise FileNotFoundError(f'OpenCV executable not found at {self.__exe_path}, please download it from {OPENCV_URL}')
		return True

	def __init__(self, opencv_exe_path: str):
		self.__exe_path = opencv_exe_path
		if REQUIRES_EXE:
			self.__check_exe_exists()

	@staticmethod
	def __get_image_at_frame_box(frame_img: np.ndarray, box: BoxData) -> np.ndarray:
		x: int
		y: int
		w: int
		h: int
		x, y, w, h = box.coords
		roi: np.ndarray = frame_img[y:y + h, x:x + w]
		return roi

	@staticmethod
	def __write_temp_image(roi: np.ndarray) -> str:
		with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_img:
			tmp_img_name: str = tmp_img.name
		cv2.imwrite(tmp_img_name, roi)
		return tmp_img_name

	def __train_image_with_tags(self,
		tmp_img_path: str,
		tags_list: str) -> None:
		cmd: List[str] = [
			self.__exe_path,
			'--train',
			'--image', tmp_img_path,
			'--tags', tags_list
		]
		subprocess.run(cmd)

	def __train_box_with_opencv(self,
	                            frame_img: np.ndarray,
	                            box: BoxData
	                            ) -> None:
		roi = self.__get_image_at_frame_box(frame_img, box)
		if roi.size == 0:
			tmp_img_name: str = self.__write_temp_image(roi)
			tags_str: str = ','.join(box.tags)
			self.__train_image_with_tags(tmp_img_name, tags_str)
			os.unlink(tmp_img_name)

	def train_boxes_with_opencv(
			self,
			frame_img: np.ndarray,
			boxes: List[BoxData]
	) -> None:
		for box in boxes:
			self.__train_box_with_opencv(
				frame_img, box
			)