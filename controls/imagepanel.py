from typing import Callable, Optional

import numpy as np
import wx

from FrameData import FrameData
from boxdata import BoxData, has_specific_identity
from controls.boxgeometry import hit_test_rect, resize_rect, rotate_rect, unrotate_rect
from events.BoxAddedEvent import BoxAddedEvent
from events.BoxEditedEvent import BoxEditedEvent
from events.BoxSelectedEvent import BoxSelectedEvent
from logutil import getLog
from pin_catalog import PinMatch

RotationAngle = int
ImageSize = tuple[int, int] # (width, height)

class ImagePanel(wx.Panel, wx.PyEventBinder):
    image: np.ndarray | None
    bitmap: wx.Bitmap | None
    __boxes: list[BoxData]
    __frame_data: FrameData
    _selected_box: BoxData | None = None
    __dragging_box: BoxData | None = None
    dragging: bool
    start_pos: wx.Point | None
    end_pos: wx.Point | None
    scale: float
    img_size: ImageSize
    bmp_size: ImageSize
    rotation_angle: RotationAngle
    resizing_box: Optional[BoxData] = None
    resize_edge: Optional[str] = None  # e.g., 'left', 'right', 'top', 'bottom', 'topleft', etc.

    _RESIZE_MARGIN: int = 6
    _CURSORS = {
        'left': wx.CURSOR_SIZEWE, 'right': wx.CURSOR_SIZEWE,
        'top': wx.CURSOR_SIZENS, 'bottom': wx.CURSOR_SIZENS,
        'topleft': wx.CURSOR_SIZENWSE, 'bottomright': wx.CURSOR_SIZENWSE,
        'topright': wx.CURSOR_SIZENESW, 'bottomleft': wx.CURSOR_SIZENESW,
    }

    def __init__(self, parent: wx.Window):
        super().__init__(parent)
        self.image = None
        self.bitmap = None
        self.__boxes = []
        self.dragging = False
        self.start_pos = None
        self.end_pos = None
        self.scale = 1.0
        self.img_size = (0, 0)
        self.bmp_size = (0, 0)
        self.rotation_angle = 0
        self.resizing_box: Optional[BoxData] = None
        self.resize_edge: Optional[str] = None  # e.g., 'left', 'right', 'top', 'bottom', 'topleft', etc.
        self._gesture_rect: tuple[int, int, int, int] | None = None
        self._gesture_anchor: tuple[int, int] | None = None
        self._selected_boxes: list[BoxData] = []
        self._box_tooltip_text: str | None = None
        self.on_confirm_box: Callable[[BoxData], None] | None = None
        self.on_confirm_boxes: Callable[[list[BoxData]], None] | None = None
        self.on_confirm_identity: Callable[[BoxData], None] | None = None
        self.on_wrong_identity: Callable[[BoxData], None] | None = None
        self.on_rank_replacements: Callable[[BoxData, int], list[PinMatch]] | None = None
        self.on_replace_pin: Callable[[BoxData, PinMatch], None] | None = None
        self.on_reject_replacements: Callable[[BoxData, list[PinMatch]], bool] | None = None
        self._context_menu_active = False
        self._replacement_popup_request: tuple[BoxData, wx.Point] | None = None
        self._context_menu_position = wx.Point(0, 0)
        self.on_box_added: Callable[[BoxData], None] | None = None
        self.on_remove_box: Callable[[BoxData], None] | None = None
        self.on_remove_boxes: Callable[[list[BoxData]], None] | None = None
        self.on_not_pin: Callable[[BoxData], None] | None = None
        self.on_not_pin_boxes: Callable[[list[BoxData]], None] | None = None
        self.on_interact: Callable[[], None] | None = None
        self.on_change_begin: Callable[[], None] | None = None
        self.on_change_end: Callable[[], None] | None = None
        self.on_undo: Callable[[], None] | None = None
        self.on_redo: Callable[[], None] | None = None
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_RIGHT_UP, self.on_right_up)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_mouse_leave)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_capture_lost)

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

    def hit_test_resize(self, pos: wx.Point) -> tuple[BoxData | None, str | None]:
        for box in reversed(self.__boxes):
            if box.review_state == 'rejected':
                continue
            edge = hit_test_rect(self._box_panel_rect(box), (pos.x, pos.y), self._RESIZE_MARGIN)
            if edge:
                return box, edge
        return None, None

    def set_image(self, img: np.ndarray, rotation_angle: int = 0) -> None:
        self._clear_box_tooltip()
        self.image = img
        self.rotation_angle = rotation_angle
        h, w = img.shape[:2]
        panel_size = self.GetSize()
        max_w, max_h = panel_size.GetWidth(), panel_size.GetHeight()
        scale = min(max_w / w, max_h / h, 1)
        new_w, new_h = int(w * scale), int(h * scale)
        import cv2
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        import wx
        wx_img = wx.Image(new_w, new_h)

        # img_rgb: np.ndarray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        # wx_img: wx.Image = wx.Image(new_w, new_h)
        # wx_img.SetData(img_rgb.tobytes())

        wx_img.SetData(img_resized.tobytes())
        self.bitmap = wx_img.ConvertToBitmap()
        self.img_size = (w, h)
        self.bmp_size = (new_w, new_h)
        self.scale = scale
        self.Refresh()

    def rotate_boxes(self, new_angle: int) -> None:
        self.rotation_angle = new_angle
        self.Refresh()

    def get_image_offset(self) -> tuple[int, int]:
        """Return the displayed bitmap's offset inside the panel."""
        panel_w, panel_h = self.GetSize().GetWidth(), self.GetSize().GetHeight()
        bmp_w, bmp_h = self.bmp_size
        return (panel_w - bmp_w) // 2, (panel_h - bmp_h) // 2

    def _source_size(self) -> tuple[int, int]:
        return (self.img_size[1], self.img_size[0]) if self.rotation_angle % 180 else self.img_size

    def _box_panel_rect(self, box: BoxData) -> tuple[int, int, int, int]:
        x, y, w, h = rotate_rect(box.coords, self._source_size(), self.rotation_angle)
        display_w, display_h = self.img_size
        bitmap_w, bitmap_h = self.bmp_size
        offset_x, offset_y = self.get_image_offset()
        left = offset_x + round(x * bitmap_w / display_w)
        top = offset_y + round(y * bitmap_h / display_h)
        right = offset_x + round((x + w) * bitmap_w / display_w)
        bottom = offset_y + round((y + h) * bitmap_h / display_h)
        return left, top, right - left, bottom - top

    def _panel_to_display(self, pos: wx.Point) -> tuple[int, int]:
        offset_x, offset_y = self.get_image_offset()
        bitmap_w, bitmap_h = self.bmp_size
        display_w, display_h = self.img_size
        return (max(0, min(display_w, round((pos.x - offset_x) * display_w / bitmap_w))),
                max(0, min(display_h, round((pos.y - offset_y) * display_h / bitmap_h))))

    def _point_on_image(self, pos: wx.Point) -> bool:
        offset_x, offset_y = self.get_image_offset()
        return (offset_x <= pos.x <= offset_x + self.bmp_size[0] and
                offset_y <= pos.y <= offset_y + self.bmp_size[1])

    def point_in_box(self, point: wx.Point, box: BoxData) -> bool:
        x, y, w, h = self._box_panel_rect(box)
        return wx.Rect(x, y, w, h).Contains(point)

    def _set_selection(self, boxes: list[BoxData]) -> None:
        unique: list[BoxData] = []
        visible_ids = {id(box) for box in self.__boxes if box.review_state != 'rejected'}
        seen: set[int] = set()
        for box in boxes:
            identity = id(box)
            if identity in visible_ids and identity not in seen:
                unique.append(box)
                seen.add(identity)
        if len(unique) == len(self._selected_boxes) and all(
                old is new for old, new in zip(self._selected_boxes, unique)):
            return
        self._selected_boxes = unique
        self._selected_box = unique[-1] if unique else None
        wx.PostEvent(self, BoxSelectedEvent(self, self._selected_box))
        self.Refresh()

    def _boxes_overlapping_panel_rect(self, rect: wx.Rect) -> list[BoxData]:
        overlapping: list[BoxData] = []
        for box in self.__boxes:
            if box.review_state == 'rejected':
                continue
            x, y, width, height = self._box_panel_rect(box)
            if width <= 0 or height <= 0:
                continue
            covered_width = max(0, min(x + width, rect.x + rect.width) - max(x, rect.x))
            covered_height = max(0, min(y + height, rect.y + rect.height) - max(y, rect.y))
            if 2 * covered_width * covered_height > width * height:
                overlapping.append(box)
        return overlapping

    def on_left_down(self, event: wx.MouseEvent) -> None:
        self._clear_box_tooltip()
        if self.on_interact:
            self.on_interact()
        if self.image is None:
            return
        self.SetFocus()
        pos = event.GetPosition()
        box, handle = self.hit_test_resize(pos)
        if box is None:
            box = self.get_box_at_position(pos)
        if getattr(event, 'ControlDown', lambda: False)():
            if box is not None:
                selected = [item for item in self._selected_boxes if item is not box]
                if len(selected) == len(self._selected_boxes):
                    selected.append(box)
                self._set_selection(selected)
            return
        if box is not None:
            if self.on_change_begin:
                self.on_change_begin()
            self._set_selection([box])
            self._gesture_rect = rotate_rect(box.coords, self._source_size(), self.rotation_angle)
            self._gesture_anchor = self._panel_to_display(pos)
            if handle:
                self.resizing_box, self.resize_edge = box, handle
            else:
                self.__dragging_box = box
            if not self.HasCapture():
                self.CaptureMouse()
            self.Refresh()
            return

        self._set_selection([])
        if self._point_on_image(pos):
            self.dragging = True
            self.start_pos = pos
            self.end_pos = pos
            if not self.HasCapture():
                self.CaptureMouse()
        self.Refresh()

    def on_right_down(self, event: wx.MouseEvent) -> None:
        self._clear_box_tooltip()
        if self.on_interact:
            self.on_interact()
        self.SetFocus()
        event.Skip()

    def on_right_up(self, event: wx.MouseEvent) -> None:
        if self.on_interact:
            self.on_interact()
        if self.image is None:
            return
        pos = event.GetPosition()
        box, _ = self.hit_test_resize(pos)
        box = box or self.get_box_at_position(pos)
        if box is not None and all(selected is not box for selected in self._selected_boxes):
            self._set_selection([box])
        if not self._selected_boxes:
            return
        menu = (self.make_selection_context_menu(list(self._selected_boxes))
                if len(self._selected_boxes) > 1 else
                self.make_box_context_menu(self._selected_boxes[0]))
        self._popup_review_menu(menu, pos)

    def _popup_review_menu(self, menu: wx.Menu, pos: wx.Point,
                           position_menu: bool = False) -> None:
        self._context_menu_active = True
        self._context_menu_position = pos
        try:
            if position_menu:
                self.PopupMenu(menu, pos)
            else:
                self.PopupMenu(menu)
        finally:
            self._context_menu_active = False
            menu.Destroy()
        request = self._replacement_popup_request
        self._replacement_popup_request = None
        if request is not None:
            wx.CallAfter(self._show_replacement_choices, *request)

    def _show_replacement_choices(self, box: BoxData, pos: wx.Point) -> None:
        if (box.review_state == 'rejected' or
                all(existing is not box for existing in self.__boxes)):
            return
        menu = self._make_replacement_menu(box)
        if menu is None:
            menu = wx.Menu()
            empty = menu.Append(wx.ID_ANY, 'No more catalog matches')
            empty.Enable(False)
        self._popup_review_menu(menu, pos, position_menu=True)

    def make_selection_context_menu(self, boxes: list[BoxData]) -> wx.Menu:
        """Apply one owner transaction to the selected visible boxes."""
        menu = wx.Menu()
        confirm = menu.Append(wx.ID_ANY, 'Confirm selected as pins')
        not_pin = menu.Append(wx.ID_ANY, '&Not a pin (automatic only)')
        remove = menu.Append(wx.ID_ANY, 'Remove selected areas\tDel')
        menu.Enable(confirm.GetId(), any(box.review_state != 'confirmed' for box in boxes))
        menu.Enable(not_pin.GetId(), any(box.source == 'automatic' for box in boxes))
        menu.Bind(wx.EVT_MENU,
                  lambda _evt: self.on_confirm_boxes(list(boxes)) if self.on_confirm_boxes else None,
                  id=confirm.GetId())
        menu.Bind(wx.EVT_MENU,
                  lambda _evt: self.on_not_pin_boxes(list(boxes)) if self.on_not_pin_boxes else None,
                  id=not_pin.GetId())
        menu.Bind(wx.EVT_MENU,
                  lambda _evt: self.on_remove_boxes(list(boxes)) if self.on_remove_boxes else None,
                  id=remove.GetId())
        return menu

    def make_box_context_menu(self, box: BoxData) -> wx.Menu:
        """Build actions bound to the same frame owner as the label buttons."""
        menu = wx.Menu()
        confirm = menu.Append(wx.ID_ANY, 'Confirm is a pin')
        confirm_identity = menu.Append(wx.ID_ANY, 'Confirm pin identity')
        wrong_identity = (menu.Append(wx.ID_ANY, 'Not this pin (&X)')
                          if has_specific_identity(box) else None)
        replacement_menu = self._make_replacement_menu(box)
        if replacement_menu is not None:
            menu.AppendSubMenu(replacement_menu, 'Replace with pin')
        remove = menu.Append(wx.ID_ANY, 'Remove area\tDel')
        not_pin = (menu.Append(wx.ID_ANY, '&Not a pin')
                   if box.source == 'automatic' else None)
        menu.Enable(confirm.GetId(), box.review_state != 'confirmed')
        menu.Enable(confirm_identity.GetId(),
                    has_specific_identity(box) and not (
                        box.review_state == 'confirmed' and box.identity_confirmed))
        menu.Bind(wx.EVT_MENU, lambda _evt: self.on_confirm_box(box) if self.on_confirm_box else None,
                  id=confirm.GetId())
        menu.Bind(wx.EVT_MENU,
                  lambda _evt: self.on_confirm_identity(box) if self.on_confirm_identity else None,
                  id=confirm_identity.GetId())
        if wrong_identity is not None:
            menu.Bind(wx.EVT_MENU,
                      lambda _evt: self.on_wrong_identity(box) if self.on_wrong_identity else None,
                      id=wrong_identity.GetId())
        menu.Bind(wx.EVT_MENU, lambda _evt: self.on_remove_box(box) if self.on_remove_box else None,
                  id=remove.GetId())
        if not_pin is not None:
            menu.Bind(wx.EVT_MENU, lambda _evt: self.on_not_pin(box) if self.on_not_pin else None,
                      id=not_pin.GetId())
        return menu

    def _make_replacement_menu(self, box: BoxData) -> wx.Menu | None:
        if self.on_rank_replacements is None:
            return None
        choices = self.on_rank_replacements(box, 20)
        if not choices:
            return None
        menu = wx.Menu()
        names = [match.name for match in choices]
        for match in choices:
            name = match.name.replace('&', '&&').replace('\t', ' ').replace('\n', ' ')
            if names.count(match.name) > 1:
                name += f" ({match.pin_id.replace('&', '&&')})"
            score = f'{match.confidence:.0%} ' if match.confidence is not None else ''
            item = menu.Append(wx.ID_ANY, score + name)
            menu.Bind(wx.EVT_MENU,
                      lambda _event, chosen=match: self.on_replace_pin(box, chosen)
                      if self.on_replace_pin else None,
                      id=item.GetId())
        menu.AppendSeparator()
        none = menu.Append(wx.ID_ANY, 'none of these')
        menu.Bind(wx.EVT_MENU,
                  lambda _event: self._reject_replacement_page(box, choices),
                  id=none.GetId())
        return menu

    def _reject_replacement_page(self, box: BoxData, choices: list[PinMatch]) -> None:
        if (self.on_reject_replacements is not None and
                self.on_reject_replacements(box, choices) and
                self._context_menu_active):
            self._replacement_popup_request = (box, self._context_menu_position)

    def get_box_at_position(self, pos: wx.Point) -> BoxData | None:
        """Get the box at the given position."""
        for box in reversed(self.__boxes):
            if box.review_state == 'rejected':
                continue
            if self.point_in_box(pos, box):
                return box
        return None

    def on_left_up(self, event: wx.MouseEvent) -> None:
        changed_gesture = self.resizing_box is not None or self.__dragging_box is not None
        if self.resizing_box is not None or self.__dragging_box is not None:
            self._update_active_box(event.GetPosition())
            self.resizing_box = None
            self.resize_edge = None
            self.__dragging_box = None
            self._gesture_rect = None
            self._gesture_anchor = None
        elif self.dragging and self.start_pos is not None:
            released = event.GetPosition()
            offset_x, offset_y = self.get_image_offset()
            end_panel = wx.Point(
                max(offset_x, min(released.x, offset_x + self.bmp_size[0])),
                max(offset_y, min(released.y, offset_y + self.bmp_size[1])))
            start_x, start_y = self._panel_to_display(self.start_pos)
            end_x, end_y = self._panel_to_display(end_panel)
            left, top = min(start_x, end_x), min(start_y, end_y)
            width, height = abs(end_x - start_x), abs(end_y - start_y)
            if width > 0 and height > 0:
                panel_left = min(self.start_pos.x, end_panel.x)
                panel_top = min(self.start_pos.y, end_panel.y)
                selection_rect = wx.Rect(panel_left, panel_top,
                                         abs(end_panel.x - self.start_pos.x),
                                         abs(end_panel.y - self.start_pos.y))
                overlapping = self._boxes_overlapping_panel_rect(selection_rect)
                if overlapping:
                    self._set_selection(overlapping)
                else:
                    if self.on_change_begin:
                        self.on_change_begin()
                    coords = unrotate_rect((left, top, width, height),
                                           self._source_size(), self.rotation_angle)
                    self.add_new_box(coords, 'user')
                    changed_gesture = True
            self.dragging = False
            self.start_pos = None
            self.end_pos = None
        if self.HasCapture():
            self.ReleaseMouse()
        if changed_gesture and self.on_change_end:
            self.on_change_end()
        self._set_resize_cursor(event.GetPosition())
        self.Refresh()
        event.Skip()

    def on_motion(self, event: wx.MouseEvent):
        pos = event.GetPosition()
        if event.LeftIsDown() and (self.resizing_box is not None or self.__dragging_box is not None):
            self._clear_box_tooltip()
            self._update_active_box(pos)
        elif event.LeftIsDown() and self.dragging:
            self._clear_box_tooltip()
            offset_x, offset_y = self.get_image_offset()
            self.end_pos = wx.Point(
                max(offset_x, min(pos.x, offset_x + self.bmp_size[0])),
                max(offset_y, min(pos.y, offset_y + self.bmp_size[1])),
            )
            self.Refresh()
        else:
            self._set_resize_cursor(pos)
            self._update_box_tooltip(pos)
        event.Skip()

    def _update_box_tooltip(self, pos: wx.Point) -> None:
        box = self.get_box_at_position(pos) if self.image is not None else None
        label = self.get_box_label_text(box) if box is not None else None
        if label == self._box_tooltip_text:
            return
        self._clear_box_tooltip()
        if label:
            tip = wx.ToolTip(label)
            tip.SetMaxWidth(480)
            self.SetToolTip(tip)
            self._box_tooltip_text = label

    def _clear_box_tooltip(self) -> None:
        if self._box_tooltip_text is not None:
            self.UnsetToolTip()
            self._box_tooltip_text = None

    def _on_mouse_leave(self, event: wx.MouseEvent) -> None:
        self._clear_box_tooltip()
        event.Skip()

    def _update_active_box(self, pos: wx.Point) -> None:
        if self._gesture_rect is None or self._gesture_anchor is None:
            return
        point = self._panel_to_display(pos)
        if self.resizing_box is not None and self.resize_edge is not None:
            box = self.resizing_box
            shown = resize_rect(self._gesture_rect, self.resize_edge, point, self.img_size)
        elif self.__dragging_box is not None:
            box = self.__dragging_box
            x, y, w, h = self._gesture_rect
            dx = point[0] - self._gesture_anchor[0]
            dy = point[1] - self._gesture_anchor[1]
            shown = (max(0, min(x + dx, self.img_size[0] - w)),
                     max(0, min(y + dy, self.img_size[1] - h)), w, h)
        else:
            return
        updated = unrotate_rect(shown, self._source_size(), self.rotation_angle)
        if updated != box.coords:
            box.coords = updated
            wx.PostEvent(self, BoxEditedEvent(source=self, box=box))
            self.Refresh()

    def _set_resize_cursor(self, pos: wx.Point) -> None:
        if self.image is None:
            return
        _, handle = self.hit_test_resize(pos)
        self.SetCursor(wx.Cursor(self._CURSORS[handle]) if handle else wx.NullCursor)

    def _on_capture_lost(self, event: wx.MouseCaptureLostEvent) -> None:
        had_gesture = self.resizing_box is not None or self.__dragging_box is not None or self.dragging
        self.resizing_box = None
        self.resize_edge = None
        self.__dragging_box = None
        self._gesture_rect = None
        self._gesture_anchor = None
        self.dragging = False
        self.start_pos = None
        self.end_pos = None
        self.SetCursor(wx.NullCursor)
        if had_gesture and self.on_change_end:
            self.on_change_end()
        self.Refresh()
        event.Skip()

    def _is_box_selected(self, box: BoxData) -> bool:
        return any(selected is box for selected in self._selected_boxes)

    @staticmethod
    def get_box_label_text(box: BoxData) -> str:
        """Show unlabelled automatic detections as review candidates."""
        name = ', '.join(box.tags or []) or box.pin_id
        if name:
            if not has_specific_identity(box):
                return 'Pin (unidentified)' if box.review_state != 'unconfirmed' else 'Candidate'
            if (not box.identity_confirmed and box.pin_id and
                    box.match_confidence is not None):
                return f'{box.match_confidence:.0%} {name}' + (
                    '?' if box.review_state != 'unconfirmed' else '')
            if box.review_state != 'unconfirmed' and not box.identity_confirmed:
                return f'{name}?'
            return name
        return 'Pin (unidentified)' if box.review_state != 'unconfirmed' else 'Candidate'

    @staticmethod
    def get_box_colour(box: BoxData) -> wx.Colour:
        if box.review_state == 'confirmed':
            return wx.Colour(0, 160, 60)
        if box.review_state == 'inherited':
            return wx.Colour(230, 140, 0)
        if has_specific_identity(box):
            return wx.RED
        return wx.BLUE

    def on_paint(self, _event: wx.PaintEvent):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        if self.bitmap:
            offset_x, offset_y = self.get_image_offset()
            dc.DrawBitmap(self.bitmap, offset_x, offset_y)
            for box in self.__boxes:
                if box.review_state != 'rejected':
                    self.paint_box(dc, box)

            if self.dragging and self.start_pos and self.end_pos:
                left, top = min(self.start_pos.x, self.end_pos.x), min(self.start_pos.y, self.end_pos.y)
                rect = wx.Rect(left, top, abs(self.end_pos.x - self.start_pos.x),
                               abs(self.end_pos.y - self.start_pos.y))
                dc.SetPen(wx.Pen(wx.BLUE, 2, wx.PENSTYLE_DOT))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRectangle(rect)

    # def to_original_image_coords(x: int, y: int) -> tuple[int, int]:
    #     bx, by = self.bmp_size
    #     iw, ih = self.img_size
    #     # Convert from bitmap to image coordinates
    #     img_x = int(x / bx * iw)
    #     img_y = int(y / by * ih)
    #     # Reverse rotation
    #     angle = self.rotation_angle
    #     if angle == 90:
    #         return img_y, iw - img_x - 1
    #     elif angle == 180:
    #         return iw - img_x - 1, ih - img_y - 1
    #     elif angle == 270:
    #         return ih - img_y - 1, img_x
    #     else:
    #         return img_x, img_y

    def undo(self):
        if self.on_undo:
            self.on_undo()

    def add_new_box(self, coords: tuple[int, int, int, int], source: str = 'user') -> None:
        """Add a new box with the given coordinates."""
        if self.on_change_begin:
            self.on_change_begin()
        new_box = BoxData(coords, [''], source,
                          review_state='confirmed' if source == 'user' else 'unconfirmed')
        self.__boxes.append(new_box)
        if self.on_box_added:
            self.on_box_added(new_box)
        box_added_event = BoxAddedEvent(self, new_box)
        getLog().info(f'New box added: {new_box}, {box_added_event}')
        wx.PostEvent(self, box_added_event)
        if self.on_change_end:
            self.on_change_end()
        self.Refresh()

    def redo(self):
        if self.on_redo:
            self.on_redo()

    def on_delete_box(self, box: BoxData) -> None:
        if self.on_remove_box:
            self.on_remove_box(box)
            return
        if box in self.__boxes:
            if self.on_change_begin:
                self.on_change_begin()
            self.__boxes.remove(box)
            if self.on_change_end:
                self.on_change_end()
            self.Refresh()  # Redraw the image panel

    def on_add_tag(self, box: BoxData, tag_number: int, tag: str) -> None:
        if box in self.__boxes and tag not in box.tags:
            if self.on_change_begin:
                self.on_change_begin()
            box.tags.append(tag)
            box.pin_id = None
            box.match_confidence = None
            box.identity_confirmed = False
            if self.on_change_end:
                self.on_change_end()
            self.Refresh()

    def on_remove_tag(self, box: BoxData, tag_number: int, tag: str) -> None:
        if box in self.__boxes and tag in box.tags:
            if self.on_change_begin:
                self.on_change_begin()
            box.tags.remove(tag)
            box.pin_id = None
            box.match_confidence = None
            box.identity_confirmed = False
            if self.on_change_end:
                self.on_change_end()
            self.Refresh()

    @property
    def boxes(self) -> list[BoxData]:
        """Get the list of boxes."""
        return self.__boxes

    @boxes.setter
    def boxes(self, new_boxes: list[BoxData]) -> None:
        """Show the current frame's shared list of boxes."""
        self._clear_box_tooltip()
        self.__boxes = new_boxes if new_boxes is not None else []
        self._set_selection(list(self._selected_boxes))
        getLog().debug(f'Setting {len(self.__boxes)} boxes in ImagePanel mutator')

        self.Refresh()  # Redraw the image panel

    def paint_box(self, dc: wx.DC, box: BoxData) -> None:
        # Check if the box is selected
        is_selected = self._selected_box is not None and self._is_box_selected(box)
        x, y, w, h = self._box_panel_rect(box)
        rect = wx.Rect(x, y, w, h)
        stroke_width = 3 if is_selected else 1

        # Draw label area at the bottom edge
        label_text = self.get_box_label_text(box) or ""
        label_rect_height = 18  # px, adjust as needed
        label_rect = wx.Rect(x, y + h - label_rect_height, w, label_rect_height)

        colour = self.get_box_colour(box)
        text_colour = wx.WHITE
        dc.SetPen(wx.Pen(colour, stroke_width))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.DrawRectangle(rect)

        # Truncate text to fit box width
        dc.SetBrush(wx.Brush(colour))
        font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        dc.SetFont(font)
        dc.SetTextForeground(text_colour)
        max_width = label_rect.GetWidth() - 4
        truncated_text = label_text
        while dc.GetTextExtent(truncated_text)[0] > max_width and len(truncated_text) > 0:
            truncated_text = truncated_text[:-1]
        if truncated_text != label_text and len(truncated_text) > 3:
            truncated_text = truncated_text[:-3] + "..."

        dc.DrawRectangle(label_rect)
        dc.DrawText(truncated_text, label_rect.x + 2, label_rect.y + 2)

    @property
    def selected_box(self) -> BoxData | None:
        """Get the currently selected box."""
        return self._selected_box

    @property
    def selected_boxes(self) -> list[BoxData]:
        """Get the current visible selection in display order."""
        return list(self._selected_boxes)
