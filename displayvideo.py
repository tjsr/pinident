import json
from typing import List

import wx

from boxdata import BoxData
# from imagescrubber import ImageScrubber
from videoscrubber import VideoScrubber

test_files: List[str] = [
    "PXL_20250715_015847092.mp4",
    "PXL_20250823_004545173.mp4",
    "PXL_20250823_004717468.mp4",
    "PXL_20250823_005120992.mp4",
]
test_file_path = "e:\\pindev"

if __name__ == '__main__':
    app = wx.App(False)
    # For images:
    # frame = ImageScrubber(None, 'Image Scrubber', 'e:\\pindev\\output')
    # For video:
    opencv_exe_path: str = 'e:\\pindev\\opencv\\build\\x64\\vc15\\bin\\opencv_video.exe'
    # "E:\\opencv\\build\\x64\\vc16\\bin"
    file_name = f'{test_file_path}\\{test_files[1]}'
    frame = VideoScrubber(None, 'Pinny Arcade video pin tagging tool', opencv_exe_path, file_name)
    frame.get_frame(1)
    frame.load_box_data()
    frame.Show()
    app.MainLoop()
