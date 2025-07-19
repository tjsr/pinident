from typing import List

import wx

from boxdata import BoxData
from controlspanel import ControlsPanel
from imagepanel import ImagePanel
from markerpanel import MarkerPanel  # Adjust import as needed
from tagpanel import TagPanel


class ScrubberFrame(wx.Frame):
    __boxes: List[BoxData] = []  # Or load from your data source
    __image_panel: ImagePanel

    @property
    def current_index(self) -> int:
        return self._current_index

    def __init__(self, parent: wx.Panel, title: str, num_frames: int):
        super().__init__(parent, title=title, size=wx.Size(800, 600))

        self._current_index = 0
        self._rotation_angle = 0
        self.num_frames = num_frames
        self.Bind(wx.EVT_SIZE, self.on_resize)

        main_panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        image_and_tag_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.__image_panel = ImagePanel(main_panel)
        # self.image_panel.Bind(EVT_BOX_ADDED, self.image_box_added)

        image_and_tag_sizer.Add(self.__image_panel, 1, wx.EXPAND | wx.ALL, 10)

        self.tag_panel = TagPanel(
            main_panel,
            self.__boxes
        )
        print(f'ScrubberFrame.__init__: TagPanel referencing {hex(id(self.__boxes))}=>{self.__boxes}')
        self.tag_panel.bind_box_events(self.__image_panel)
        # self.image_panel.Bind(EVT_BOX_ADDED, self.tag_panel.Refresh)

        image_and_tag_sizer.Add(self.tag_panel, 0, wx.EXPAND | wx.ALL, 10)

        vbox.Add(image_and_tag_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Add ControlsPanel below image_panel
        button_panel = ControlsPanel(main_panel)
        vbox.Add(button_panel, 0, wx.CENTER, 0)
        button_panel.bind_buttons(self.on_prev, self.on_next, self.on_rotate_ccw, self.on_rotate_cw)

        # Add MarkerPanel below image_panel
        self.marker_panel = MarkerPanel(main_panel, num_frames)
        vbox.Add(self.marker_panel, 0, wx.EXPAND | wx.ALL, 5)

        self.slider = wx.Slider(main_panel, value=0, minValue=0, maxValue=max(0, num_frames-1),
                                style=wx.SL_HORIZONTAL | wx.SL_LABELS)
        self.slider.Bind(wx.EVT_SLIDER, self.on_slider)
        vbox.Add(self.slider, 0, wx.EXPAND | wx.ALL, 10)

        main_panel.SetSizer(vbox)

        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_down)
        self.Bind(wx.EVT_SHOW, self.on_show)

    def get_frame(self, index: int, rotation_angle: int = 0):
        raise NotImplementedError

    def display_image(self):
        img = self.get_frame(self._current_index, self._rotation_angle)
        if img is None:
            return
        panel_size = self.__image_panel.GetSize()
        if panel_size.GetWidth() < 10 or panel_size.GetHeight() < 10:
            return  # Panel not yet sized, skip

        self.__image_panel.set_image(img, self._rotation_angle)
        self.slider.SetValue(self._current_index)
        self.Refresh()

    def on_resize(self, event):
        self.display_image()
        event.Skip()

    def on_prev(self, event):
        if self._current_index > 0:
            self._current_index -= 1
            self.display_image()

    def on_next(self, event):
        if self._current_index < self.num_frames - 1:
            self._current_index += 1
            self.display_image()

    def on_slider(self, event):
        self._current_index = self.slider.GetValue()
        self.display_image()

    def on_rotate_cw(self, event: wx.CommandEvent) -> None:
        self._rotation_angle = (self._rotation_angle + 90) % 360
        self.__image_panel.rotate_boxes(self._rotation_angle)
        self.display_image()

    def on_rotate_ccw(self, event: wx.CommandEvent) -> None:
        self._rotation_angle = (self._rotation_angle - 90) % 360
        self.__image_panel.rotate_boxes(self._rotation_angle)
        self.display_image()

    def on_show(self, event):
        if event.IsShown():
            self.GetChildren()[0].Layout()  # main_panel.Layout()
            self.display_image()
        event.Skip()

    def on_key_down(self, event):
        keycode = event.GetKeyCode()
        control_down = event.ControlDown()
        shift_down = event.ShiftDown()
        # Ctrl+Z for undo
        if control_down and keycode == ord('Z') and not shift_down:
            self.__image_panel.undo()
        # Ctrl+Shift+Z for redo
        elif control_down and keycode == ord('Z') and shift_down:
            self.__image_panel.redo()
        else:
            event.Skip()

    def on_box_update(self) -> None:
        """Called when a box is updated, e.g., after adding or removing a tag."""
        self.tag_panel.update_boxes(self.__boxes)
        self.Refresh()

    @current_index.setter
    def current_index(self, index):
        if 0 <= index < self.num_frames:
            self._current_index = index
            if self.slider is not None:
                self.slider.SetValue(index)
                self.display_image()
        else:
            raise ValueError("Index out of bounds")
