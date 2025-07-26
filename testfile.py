import cv2

from logutil import getLog

file_name: str = "e:\\pindev\\PXL_20250715_015847092.mp4"
out_dir: str = "e:\\pindev\\output"
import os

log = getLog()

try:
	# creating a folder named data
	if not os.path.exists(out_dir):
		os.makedirs(out_dir)

# if not created then raise error
except OSError as e:
	log.error('Error: Creating directory of data')
	raise e

existing_files = [f for f in os.listdir(out_dir) if os.path.isfile(os.path.join(out_dir, f))]

video_cap = cv2.VideoCapture(file_name)
total_frames: int = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames in the video
fps: float = video_cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
resolution: tuple = (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
if not video_cap.isOpened():
	log.error("Error: Could not open video.")
	raise Exception("Could not open video file.")

log.info(f'Video is at {resolution[0]}x{resolution[1]} @ {fps}fps with {total_frames} frames.')

success, image = video_cap.read()
display_image = None
selected_frame: int = 115
frame_number = 0
max_output_frames: int = -1
while success and (max_output_frames < 0 or frame_number < max_output_frames):  # Limit to 10 frames for testing
	output_file: str = os.path.join(out_dir, f"frame{frame_number}.jpg")
	if not output_file in existing_files:
	# if not os.path.exists(output_file):
		# print('Writing new frame: ', success)
		cv2.imwrite(output_file, image)  # save frame as JPEG file

	if frame_number == selected_frame:
		display_image = image.copy()

	success,image = video_cap.read()
	frame_number += 1

cv2.imshow('Frame', display_image)
key: int = cv2.waitKey(0)
cv2.destroyAllWindows()
#
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()