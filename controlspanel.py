import wx


class ControlsPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        sizer: wx.BoxSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.prev_btn = wx.Button(self, label='Previous')
        self.next_btn = wx.Button(self, label='Next')
        self.next_empty_button = wx.Button(self, label="Next Empty")
        self.rotate_ccw_btn = wx.Button(self, label='Rotate CCW')
        self.rotate_cw_btn = wx.Button(self, label='Rotate CW')
        self.remove_selected_btn = wx.Button(self, label='Remove Selected')

        sizer.Add(self.prev_btn, 0, wx.ALL, 5)
        sizer.Add(self.next_btn, 0, wx.ALL, 5)
        sizer.Add(self.next_empty_button, 0, wx.ALL, 5)
        sizer.Add(self.rotate_ccw_btn, 0, wx.ALL, 5)
        sizer.Add(self.rotate_cw_btn, 0, wx.ALL, 5)
        sizer.Add(self.remove_selected_btn, 0, wx.ALL, 5)
        self.SetSizer(sizer)

    def bind_buttons(self, prev_handler, next_handler, on_next_empty, rotate_ccw_handler, rotate_cw_handler, on_remove_selected):
        self.prev_btn.Bind(wx.EVT_BUTTON, prev_handler)
        self.next_btn.Bind(wx.EVT_BUTTON, next_handler)
        self.next_empty_button.Bind(wx.EVT_BUTTON, on_next_empty)
        self.rotate_ccw_btn.Bind(wx.EVT_BUTTON, rotate_ccw_handler)
        self.rotate_cw_btn.Bind(wx.EVT_BUTTON, rotate_cw_handler)
        self.remove_selected_btn.Bind(wx.EVT_BUTTON, on_remove_selected)

    def set_prev_enabled(self, enabled: bool):
        self.prev_btn.Enable(enabled)

    def set_next_enabled(self, enabled: bool):
        self.next_btn.Enable(enabled)
