import wx

from imagescrubber import ImageScrubber
from videoscrubber import VideoScrubber

file_name: str = "e:\\pindev\\PXL_20250715_015847092.mp4"

if __name__ == '__main__':
    app = wx.App(False)
    # For images:
    # frame = ImageScrubber(None, 'Image Scrubber', 'e:\\pindev\\output')
    # For video:
    frame = VideoScrubber(None, 'Pinny Arcade video pin tagging tool', file_name)
    frame.get_frame(1)
    frame.Show()
    app.MainLoop()
