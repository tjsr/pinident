import wx


class ControlsPanel(wx.Panel):
    def __init__(self, parent):
        super().__init__(parent)
        sizer: wx.BoxSizer = wx.BoxSizer(wx.VERTICAL)
        actions = wx.BoxSizer(wx.HORIZONTAL)
        self.prev_btn = wx.Button(self, label='Previous')
        self.next_btn = wx.Button(self, label='Next')
        self.play_btn = wx.Button(self, label='Play')

        self.next_empty_button = wx.Button(self, label="Next Empty")
        self.rotate_ccw_btn = wx.Button(self, label='Rotate CCW')
        self.rotate_cw_btn = wx.Button(self, label='Rotate CW')
        self.remove_selected_btn = wx.Button(self, label='Remove Selected')
        self.process_btn = wx.Button(self, label='Detect Pins')
        self.confirm_all_btn = wx.Button(self, label='Confirm all suggestions')
        self.confirm_all_btn.SetToolTip(
            'Confirm identified suggestions on this frame; leave unidentified areas unchanged.')

        actions.Add(self.prev_btn, 0, wx.ALL, 5)
        actions.Add(self.next_btn, 0, wx.ALL, 5)
        actions.Add(self.play_btn, 0, wx.ALL, 5)
        actions.Add(self.next_empty_button, 0, wx.ALL, 5)
        actions.Add(self.rotate_ccw_btn, 0, wx.ALL, 5)
        actions.Add(self.rotate_cw_btn, 0, wx.ALL, 5)
        actions.Add(self.remove_selected_btn, 0, wx.ALL, 5)
        actions.Add(self.process_btn, 0, wx.ALL, 5)
        sizer.Add(actions, 0, wx.ALIGN_CENTER_HORIZONTAL)
        sizer.Add(self.confirm_all_btn, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.BOTTOM, 5)
        self.SetSizer(sizer)

    def bind_buttons(self, prev_handler, next_handler, on_next_empty, rotate_ccw_handler, rotate_cw_handler, on_remove_selected, on_process, play_handler, confirm_all_handler):
        self.prev_btn.Bind(wx.EVT_BUTTON, prev_handler)
        self.next_btn.Bind(wx.EVT_BUTTON, next_handler)
        self.play_btn.Bind(wx.EVT_BUTTON, play_handler)

        self.next_empty_button.Bind(wx.EVT_BUTTON, on_next_empty)
        self.rotate_ccw_btn.Bind(wx.EVT_BUTTON, rotate_ccw_handler)
        self.rotate_cw_btn.Bind(wx.EVT_BUTTON, rotate_cw_handler)
        self.remove_selected_btn.Bind(wx.EVT_BUTTON, on_remove_selected)
        self.process_btn.Bind(wx.EVT_BUTTON, on_process)
        self.confirm_all_btn.Bind(wx.EVT_BUTTON, confirm_all_handler)

    def set_prev_enabled(self, enabled: bool):
        self.prev_btn.Enable(enabled)

    def set_next_enabled(self, enabled: bool):
        self.next_btn.Enable(enabled)
