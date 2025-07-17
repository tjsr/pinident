import wx


class ControlsPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        sizer: wx.BoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.prev_btn = wx.Button(self, label='Previous')
        self.next_btn = wx.Button(self, label='Next')
        self.rotate_ccw_btn = wx.Button(self, label='Rotate CCW')
        self.rotate_cw_btn = wx.Button(self, label='Rotate CW')
        sizer.Add(self.prev_btn, 0, wx.ALL, 5)
        sizer.Add(self.next_btn, 0, wx.ALL, 5)
        sizer.Add(self.rotate_ccw_btn, 0, wx.ALL, 5)
        sizer.Add(self.rotate_cw_btn, 0, wx.ALL, 5)
        self.SetSizer(sizer)

    def bind_buttons(self, prev_handler, next_handler, rotate_ccw_handler, rotate_cw_handler):
        self.prev_btn.Bind(wx.EVT_BUTTON, prev_handler)
        self.next_btn.Bind(wx.EVT_BUTTON, next_handler)
        self.rotate_ccw_btn.Bind(wx.EVT_BUTTON, rotate_ccw_handler)
        self.rotate_cw_btn.Bind(wx.EVT_BUTTON, rotate_cw_handler)
