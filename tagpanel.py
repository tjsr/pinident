import wx
from typing import Callable, List

from boxdata import BoxData
from boxtagpanel import BoxTagPanelEdit


class TagPanel(wx.Panel):
    boxes: List[BoxData]
    __box_panels: List[BoxTagPanelEdit] = []
    # on_delete_box: Callable[[BoxData], None]
    on_add_tag: Callable[[BoxData, int, str], None]
    on_remove_tag: Callable[[BoxData, int, str], None]
    __on_box_updated: Callable[[BoxData], None]
    __on_boxes_updated: Callable[[], None]

    vbox: wx.BoxSizer
    box_sizers: List[wx.BoxSizer]
    tag_sizers: List[wx.BoxSizer]
    pin_cb: wx.CheckBox
    set_cb: wx.CheckBox
    card_cb: wx.CheckBox

    def __init__(
        self,
        parent: wx.Window,
        boxes: List[BoxData],
        on_delete_box: Callable[[BoxData], None],
        on_add_tag: Callable[[BoxData, int, str], None],
        on_remove_tag: Callable[[BoxData, int, str], None]
    ) -> None:
        super().__init__(parent)
        self.boxes = boxes
        self.on_delete_box = on_delete_box
        self.on_add_tag = on_add_tag
        self.on_remove_tag = on_remove_tag

        self.vbox = wx.BoxSizer(wx.VERTICAL)
        self.box_sizers = []
        self.heading_texts = []
        self.del_buttons = []
        self.tag_sizers = []
        self.combo_boxes = []
        self.rem_buttons = []
        self.add_buttons = []

        self.pin_cb = wx.CheckBox(self, label="Contains pin")
        self.set_cb = wx.CheckBox(self, label="Contains set")
        self.card_cb = wx.CheckBox(self, label="With backing card")

        self.vbox.Add(self.pin_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.set_cb, 0, wx.ALL, 5)
        self.vbox.Add(self.card_cb, 0, wx.ALL, 5)

        self.set_ui()
        self.SetSizer(self.vbox)

    def set_ui(self) -> None:
        # Remove all box/tag controls but keep checkboxes
        for sizer in self.box_sizers:
            self.vbox.Remove(sizer)
        self.box_sizers.clear()
        self.heading_texts.clear()
        self.del_buttons.clear()
        self.tag_sizers.clear()
        self.combo_boxes.clear()
        self.rem_buttons.clear()
        self.add_buttons.clear()

        while len(self.__box_panels) > 0:
            panel = self.__box_panels.pop()
            panel.Destroy()

        for idx, box in enumerate(self.boxes):
            box_tag_panel = BoxTagPanelEdit(self, box)
            self.__box_panels.append(box_tag_panel)
            box_sizer = wx.BoxSizer(wx.VERTICAL)
            box_sizer.Add(box_tag_panel, 0, wx.EXPAND)

            # coords: Tuple[int, int, int, int] = box['coords']
            # heading = wx.StaticText(self, label=f"Box {idx+1}: x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}")
            # del_btn = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_BUTTON))
            # del_btn.Bind(wx.EVT_BUTTON, lambda evt, i=idx: self.on_delete_box(i))
            # heading_sizer = wx.BoxSizer(wx.HORIZONTAL)
            # heading_sizer.Add(heading, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            # heading_sizer.Add(del_btn, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)
            # box_sizer.Add(heading_sizer, 0, wx.EXPAND)
            #
            # tag_sizer = wx.BoxSizer(wx.HORIZONTAL)
            # combo_list: List[wx.ComboBox] = []
            # rem_btn_list: List[wx.BitmapButton] = []
            # for t_idx, tag in enumerate(box['tags']):
            #     combo = wx.ComboBox(self, value=tag, choices=[], style=wx.CB_DROPDOWN)
            #     combo.Bind(wx.EVT_TEXT, lambda evt, i=idx, ti=t_idx: self.on_add_tag(i, ti, evt.GetString()))
            #     combo_list.append(combo)
            #     tag_sizer.Add(combo, 0, wx.ALL, 2)
            #     rem_btn = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_BUTTON))
            #     rem_btn.Bind(wx.EVT_BUTTON, lambda evt, i=idx, ti=t_idx: self.on_remove_tag(i, ti))
            #     rem_btn_list.append(rem_btn)
            #     tag_sizer.Add(rem_btn, 0, wx.ALL, 2)
            # add_btn = wx.BitmapButton(self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_PLUS, wx.ART_BUTTON))
            # add_btn.Bind(wx.EVT_BUTTON, lambda evt, i=idx: self.on_add_tag(i, len(box['tags']), ""))
            # tag_sizer.Add(add_btn, 0, wx.ALL, 2)
            # box_sizer.Add(tag_sizer, 0, wx.EXPAND)
            #
            self.box_sizers.append(box_sizer)
            # self.heading_texts.append(heading)
            # self.del_buttons.append(del_btn)
            # self.tag_sizers.append(tag_sizer)
            # self.combo_boxes.append(combo_list)
            # self.rem_buttons.append(rem_btn_list)
            # self.add_buttons.append(add_btn)

            self.vbox.Add(box_sizer, 0, wx.EXPAND | wx.ALL, 5)
        self.Layout()

    def update_boxes(self, boxes: List[BoxData]) -> None:
        self.boxes = boxes
        self.set_ui()
