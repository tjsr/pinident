"""Editable pin label with a debounced catalog popup."""

from collections import Counter
from heapq import nlargest

import wx

from boxdata import BoxData
from events import BoxLabelRemoveEvent
from events.BoxLabelEditEvent import BoxLabelEditedEvent
from logutil import getLog


class BoxTagLabelRow(wx.Panel):
    MAX_VISIBLE_CHOICES = 75
    SEARCH_DELAY_MS = 240

    def __init__(
        self,
        parent: wx.Panel,
        box: BoxData,
        tag_index: int,
        catalog_options: list[tuple[str, str]] | None = None,
    ):
        super().__init__(parent)
        label = box.get_tag(tag_index) if tag_index < len(box.tags or []) else ''
        self.__tag_index = tag_index
        self.__is_only_row = False
        self._catalog_by_display: dict[str, tuple[str, str]] = {}
        self._display_by_catalog: dict[tuple[str, str], str] = {}
        self._visible_catalog_labels: list[str] = []
        self._popup: wx.PopupWindow | None = None
        self._results: wx.ListBox | None = None

        self.__text_entry = wx.TextCtrl(self, value=label)
        self.__text_entry.SetMinSize(wx.Size(280, -1))
        self.__text_entry.SetHint('Type to search pins')
        self.__search_button = wx.Button(self, label='\u25be',
                                          size=wx.Size(26, -1), style=wx.BU_EXACTFIT)
        self.__search_button.SetToolTip('Show pin choices')
        self.__rem_button = wx.BitmapButton(
            self, bitmap=wx.ArtProvider.GetBitmap(wx.ART_MINUS, wx.ART_BUTTON))
        self.timer = wx.Timer(self)
        self._search_timer = wx.Timer(self)

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(self.__text_entry, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(self.__search_button, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 3)
        sizer.Add(self.__rem_button, 0, wx.ALL, 5)
        self.SetSizer(sizer)

        self.__text_entry.Bind(wx.EVT_TEXT, self.label_edited)
        self.__text_entry.Bind(wx.EVT_KEY_DOWN, self._on_text_key)
        self.__text_entry.Bind(wx.EVT_KILL_FOCUS, self._on_picker_focus_lost)
        self.__search_button.Bind(wx.EVT_BUTTON, self._open_catalog)
        self.__search_button.Bind(wx.EVT_KILL_FOCUS, self._on_picker_focus_lost)
        self.__rem_button.Bind(wx.EVT_BUTTON, self.on_remove_button_clicked)
        self.Bind(wx.EVT_TIMER, self.on_label_edited_timer, self.timer)
        self.Bind(wx.EVT_TIMER, self._on_search_timer, self._search_timer)
        self.Bind(wx.EVT_WINDOW_DESTROY, self.__on_text_destroy)
        self.set_catalog_options(catalog_options or [])

    @property
    def tag(self) -> str:
        return self.__text_entry.GetValue()

    @property
    def visible_options(self) -> list[str]:
        return list(self._visible_catalog_labels)

    def set_catalog_options(self, options: list[tuple[str, str]]) -> None:
        name_counts = Counter(name for _pin_id, name in options)
        self._catalog_by_display = {
            (name if name_counts[name] == 1 else f'{name} [{pin_id}]'): (pin_id, name)
            for pin_id, name in options}
        self._display_by_catalog = {
            option: display for display, option in self._catalog_by_display.items()}
        self.__text_entry.SetToolTip(
            self.tag or f'Type to search {len(self._catalog_by_display)} pins')
        if self._popup is not None and self._popup.IsShown():
            self._show_matches(self.tag)

    def _matching_labels(self, query: str) -> list[str]:
        needle = query.strip().casefold()
        return [label for label, (pin_id, _name) in self._catalog_by_display.items()
                if not needle or needle in label.casefold() or needle in pin_id.casefold()
                ][:self.MAX_VISIBLE_CHOICES]

    def _ensure_popup(self) -> None:
        if self._popup is not None:
            return
        # A transient popup can dismiss the first list click while the editor holds focus.
        self._popup = wx.PopupWindow(self, wx.BORDER_SIMPLE)
        self._results = wx.ListBox(self._popup)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self._results, 1, wx.EXPAND)
        self._popup.SetSizer(sizer)
        self._results.Bind(wx.EVT_LISTBOX, self.on_catalog_selected)
        self._results.Bind(wx.EVT_LEFT_DOWN, self._on_result_mouse_down)
        self._results.Bind(wx.EVT_KILL_FOCUS, self._on_picker_focus_lost)

    def _on_result_mouse_down(self, event: wx.MouseEvent) -> None:
        if self._results is not None:
            index = self._results.HitTest(event.GetPosition())
            if index != wx.NOT_FOUND:
                self.select_catalog_option(index)
                return
        event.Skip()

    def _on_picker_focus_lost(self, event: wx.FocusEvent) -> None:
        wx.CallAfter(self._hide_popup_if_unfocused)
        event.Skip()

    def _hide_popup_if_unfocused(self) -> None:
        if self._popup is None or not self._popup.IsShown():
            return
        focus = wx.Window.FindFocus()
        while focus is not None:
            if focus in (self.__text_entry, self.__search_button, self._popup):
                return
            focus = focus.GetParent()
        self._popup.Hide()

    def _show_matches(self, query: str) -> None:
        labels = self._matching_labels(query)
        self._visible_catalog_labels = labels
        if not labels or self.IsBeingDeleted():
            if self._popup is not None and self._popup.IsShown():
                self._popup.Hide()
            return
        self._ensure_popup()
        assert self._popup is not None and self._results is not None
        self._results.SetItems(labels)

        dc = wx.ClientDC(self.__text_entry)
        dc.SetFont(self.__text_entry.GetFont())
        text_width = max(dc.GetTextExtent(label).width
                         for label in nlargest(12, labels, key=len)) + 36
        width = min(700, max(self.__text_entry.GetSize().width, text_width))
        height = min(300, 24 * min(len(labels), 10) + 4)
        self._popup.SetSize(wx.Size(width, height))
        self._popup.Layout()

        origin = self.__text_entry.ClientToScreen(
            wx.Point(0, self.__text_entry.GetSize().height))
        display_index = wx.Display.GetFromWindow(self)
        area = wx.Display(display_index if display_index >= 0 else 0).GetClientArea()
        x = max(area.x, min(origin.x, area.GetRight() - width))
        y = (origin.y if origin.y + height <= area.GetBottom() else
             origin.y - self.__text_entry.GetSize().height - height)
        self._popup.Move(wx.Point(x, max(area.y, y)))
        if not self._popup.IsShown():
            self._popup.Show()

    def _open_catalog(self, _event: wx.CommandEvent) -> None:
        self._search_timer.Stop()
        self._show_matches(self.tag)

    def _on_search_timer(self, _event: wx.TimerEvent) -> None:
        if self.__text_entry.HasFocus():
            self._show_matches(self.tag)

    def _on_text_key(self, event: wx.KeyEvent) -> None:
        key = event.GetKeyCode()
        if key in (wx.WXK_DOWN, wx.WXK_UP):
            if self._popup is None or not self._popup.IsShown():
                self._search_timer.Stop()
                self._show_matches(self.tag)
            elif self._results is not None and self._results.GetCount():
                current = self._results.GetSelection()
                step = 1 if key == wx.WXK_DOWN else -1
                self._results.SetSelection(
                    (current + step) % self._results.GetCount())
            return
        if key == wx.WXK_ESCAPE and self._popup is not None and self._popup.IsShown():
            self._popup.Hide()
            return
        if key == wx.WXK_RETURN and self._popup is not None and self._popup.IsShown():
            assert self._results is not None
            self.select_catalog_option(max(0, self._results.GetSelection()))
            return
        event.Skip()

    def select_catalog_option(self, index: int) -> None:
        if not 0 <= index < len(self._visible_catalog_labels):
            return
        display = self._visible_catalog_labels[index]
        self.timer.Stop()
        self._search_timer.Stop()
        self.__text_entry.ChangeValue(display)
        self.__text_entry.SetToolTip(display)
        if self._popup is not None and self._popup.IsShown():
            self._popup.Hide()
        self.fire_edited_event()

    def on_catalog_selected(self, _event: wx.CommandEvent) -> None:
        if self._results is not None:
            self.select_catalog_option(self._results.GetSelection())

    def repaint(self) -> None:
        if self.__is_only_row:
            self.__rem_button.Hide()
        else:
            self.__rem_button.Show()

    def set_only_row(self, is_only_row: bool) -> None:
        self.__is_only_row = is_only_row
        self.repaint()

    def set_tag(self, tag: str, is_only_row: bool, pin_id: str | None = None) -> None:
        if self.__text_entry.IsBeingDeleted():
            return
        display = self._display_by_catalog.get((pin_id, tag), tag)
        if display is not None and display != self.tag:
            self.timer.Stop()
            self._search_timer.Stop()
            self.__text_entry.ChangeValue(display)
            self.__text_entry.SetToolTip(display)
            if self._popup is not None and self._popup.IsShown():
                self._show_matches(display)
        if is_only_row != self.__is_only_row:
            self.set_only_row(is_only_row)

    def label_edited(self, _event: wx.Event) -> None:
        self._search_timer.Start(self.SEARCH_DELAY_MS, oneShot=True)
        self.timer.Start(500, oneShot=True)

    def on_label_edited_timer(self, event: wx.TimerEvent) -> None:
        self.fire_edited_event()
        event.Skip()

    def fire_edited_event(self) -> None:
        value = self.tag
        selected = self._catalog_by_display.get(value)
        wx.PostEvent(self, BoxLabelEditedEvent(
            self, self.__tag_index, selected[1] if selected else value,
            selected[0] if selected else None))

    def on_remove_button_clicked(self, event: wx.CommandEvent) -> None:
        if self.__is_only_row:
            getLog().debug('Cannot remove the only row.')
            return
        wx.PostEvent(self, BoxLabelRemoveEvent(event.GetEventObject(), self.__tag_index))

    def __on_text_destroy(self, event: wx.WindowDestroyEvent) -> None:
        if event.GetEventObject() is self:
            self.timer.Stop()
            self._search_timer.Stop()
        event.Skip()
