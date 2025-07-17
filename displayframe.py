import wx
import cv2
import os
# import numpy as np

class ImageScrubber(wx.Frame):
    def __init__(self, parent, title, image_dir):
        super().__init__(parent, title=title, size=(800, 600))
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')]
        self.image_files.sort()
        self.current_index = 0
        self.rotation_angle = 0  # Track rotation in degrees
        self.Bind(wx.EVT_SIZE, self.on_resize)

        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        self.image_panel = wx.Panel(panel)
        image_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.image_ctrl = wx.StaticBitmap(self.image_panel)
        image_panel_sizer.AddStretchSpacer(1)
        image_panel_sizer.Add(self.image_ctrl, 0, wx.ALIGN_CENTER_VERTICAL)
        image_panel_sizer.AddStretchSpacer(1)
        self.image_panel.SetSizer(image_panel_sizer)
        vbox.Add(self.image_panel, 1, wx.EXPAND | wx.ALL, 10)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        prev_btn = wx.Button(panel, label='Previous')
        next_btn = wx.Button(panel, label='Next')
        rotate_cw_btn = wx.Button(panel, label='Rotate CW')
        rotate_ccw_btn = wx.Button(panel, label='Rotate CCW')
        hbox.Add(prev_btn, 0, wx.ALL, 5)
        hbox.Add(next_btn, 0, wx.ALL, 5)
        hbox.Add(rotate_cw_btn, 0, wx.ALL, 5)
        hbox.Add(rotate_ccw_btn, 0, wx.ALL, 5)
        vbox.Add(hbox, 0, wx.CENTER)

        # Add marker panel above slider
        self.marker_panel = MarkerPanel(panel, len(self.image_files), interval=5)
        vbox.Add(self.marker_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 10)

        # Add horizontal slider
        self.slider = wx.Slider(panel, value=0, minValue=0, maxValue=max(0, len(self.image_files)-1),
                                style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        vbox.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)

        panel.SetSizer(vbox)

        prev_btn.Bind(wx.EVT_BUTTON, self.on_prev)
        next_btn.Bind(wx.EVT_BUTTON, self.on_next)
        self.slider.Bind(wx.EVT_SLIDER, self.on_slider)
        rotate_cw_btn.Bind(wx.EVT_BUTTON, self.on_rotate_cw)
        rotate_ccw_btn.Bind(wx.EVT_BUTTON, self.on_rotate_ccw)

        self.display_image()
        self.Bind(wx.EVT_SHOW, self.on_show)

    def display_image(self):
        if not self.image_files:
            return
        img_path = os.path.join(self.image_dir, self.image_files[self.current_index])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Rotate image based on current angle
        if self.rotation_angle % 360 == 90:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotation_angle % 360 == 180:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif self.rotation_angle % 360 == 270:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

        h, w = img.shape[:2]
        panel_size = self.image_panel.GetSize()
        max_w, max_h = panel_size.GetWidth(), panel_size.GetHeight()
        scale = min(max_w / w, max_h / h, 1)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        wx_img = wx.Image(new_w, new_h)
        wx_img.SetData(img_resized.tobytes())
        bmp = wx_img.ConvertToBitmap()
        self.image_ctrl.SetBitmap(bmp)
        self.slider.SetValue(self.current_index)
        self.Refresh()

    def on_resize(self, event):
        self.display_image()
        event.Skip()

    def on_prev(self, event):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_image()

    def on_next(self, event):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_image()

    def on_slider(self, event):
        self.current_index = self.slider.GetValue()
        self.display_image()

    def on_rotate_cw(self, event):
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.display_image()

    def on_rotate_ccw(self, event):
        self.rotation_angle = (self.rotation_angle - 90) % 360
        self.display_image()

    def on_show(self, event):
        if event.IsShown():
            self.display_image()
        event.Skip()

if __name__ == '__main__':
    app = wx.App(False)
    frame = ImageScrubber(None, 'Image Scrubber', 'e:\\pindev\\output')
    frame.Show()
    app.MainLoop()

