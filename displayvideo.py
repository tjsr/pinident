from logutil import install_exception_logging

def main() -> None:
    install_exception_logging()
    import wx

    from videoscrubber import VideoScrubber
    from pin_catalog import PinCatalog
    from pinpanion_sync import start_catalog_sync

    app = wx.App(False)
    with wx.FileDialog(None, 'Open a pin video or image',
                       wildcard=('Video and image files|*.mp4;*.avi;*.mov;*.mkv;'
                                 '*.png;*.jpg;*.jpeg;*.bmp;*.webp;*.tif;*.tiff|'
                                 'All files|*.*'),
                       style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
        if dialog.ShowModal() != wx.ID_OK:
            return
        video_path = dialog.GetPath()

    frame = VideoScrubber(None, 'Pinny Arcade pin tagging tool', '',
                          video_path=video_path, load_catalog=False)
    frame.load_box_data()
    frame.Show()

    def model_ready(matcher):
        def install():
            try:
                frame.set_pin_matcher(matcher)
            except RuntimeError:
                pass  # The video window was closed while the background sync ran.
        wx.CallAfter(install)

    start_catalog_sync(PinCatalog(), model_ready)
    app.MainLoop()


if __name__ == '__main__':
    main()
