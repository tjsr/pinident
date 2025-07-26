import numpy as np
import wx
import copy

from boxdata import BoxData
from events.BoxAddedEvent import BoxAddedEvent
from events.BoxSelectedEvent import BoxSelectedEvent, BoxDeselectedEvent
from events.BoxUpdatedEvent import BoxUpdatedEvent
from events.events import wxEVT_BOX_SELECTED, EVT_BOX_SELECTED
from logutil import getLog

RotationAngle = int
ImageSize = tuple[int, int] # (width, height)
UserAction = str  # 'draw_box', 'rotate', etc.

class ImagePanel(wx.Panel, wx.PyEventBinder):
    image: np.ndarray | None
    bitmap: wx.Bitmap | None
    __boxes: list[BoxData]
    _selected_box: BoxData | None = None
    dragging: bool
    start_pos: wx.Point | None
    end_pos: wx.Point | None
    scale: float
    img_size: ImageSize
    bmp_size: ImageSize
    rotation_angle: RotationAngle
    undo_stack: list[tuple[UserAction, RotationAngle, list[BoxData]] | tuple[UserAction, list[BoxData]]]
    redo_stack: list[tuple[UserAction, RotationAngle, list[BoxData]] | tuple[UserAction, list[BoxData]]]

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
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self.on_left_down)
        self.Bind(wx.EVT_LEFT_UP, self.on_left_up)
        self.Bind(wx.EVT_MOTION, self.on_motion)
        self.undo_stack = []
        self.redo_stack = []

        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)

    def set_image(self, img: np.ndarray, rotation_angle: int = 0) -> None:
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
        # Store undo state
        self.undo_stack.append(('rotate', self.rotation_angle, copy.deepcopy(self.__boxes)))
        self.redo_stack.clear()

        # Only update the rotation angle
        self.rotation_angle = new_angle

        # Trigger box update event and refresh
        update_event = BoxUpdatedEvent(self, self.__boxes)
        wx.PostEvent(self, update_event)
        self.Refresh()

    def get_image_offset(self) -> tuple[int, int]:
        """Return the (x, y) offset of the image inside the panel, accounting for rotation."""
        panel_w: int = self.GetSize().GetWidth()
        panel_h: int = self.GetSize().GetHeight()
        bmp_w, bmp_h = self.bmp_size
        if self.rotation_angle in (90, 270):
            bmp_w, bmp_h = bmp_h, bmp_w
        offset_x: int = (panel_w - bmp_w) // 2
        offset_y: int = (panel_h - bmp_h) // 2
        return offset_x, offset_y

    def clamp_to_image(self, x: int, y: int) -> tuple[int, int]:
        """Clamp coordinates to the image area."""
        img_w: int = self.bmp_size[0]
        img_h: int = self.bmp_size[1]
        x_clamped: int = max(0, min(x, img_w - 1))
        y_clamped: int = max(0, min(y, img_h - 1))
        return x_clamped, y_clamped


    def point_in_box(self, point: wx.Point, box: BoxData) -> bool:
        """Check if a wx.Point is inside the box (in bitmap coordinates)."""
        x, y, w, h = box.coords
        # Map image coordinates to bitmap coordinates
        img_w, img_h = self.img_size
        bx, by = self.bmp_size
        if self.rotation_angle in (90, 270):
            img_w, img_h = img_h, img_w
            bx, by = by, bx
        bmp_x1 = int(x / img_w * bx)
        bmp_y1 = int(y / img_h * by)
        bmp_x2 = int((x + w) / img_w * bx)
        bmp_y2 = int((y + h) / img_h * by)
        rect = wx.Rect(bmp_x1, bmp_y1, bmp_x2 - bmp_x1, bmp_y2 - bmp_y1)
        return rect.Contains(point)

    def on_left_down(self, event: wx.MouseEvent) -> None:
        mouse_pos: wx.Point = event.GetPosition()
        offset_x, offset_y = self.get_image_offset()
        img_x: int = mouse_pos.x - offset_x
        img_y: int = mouse_pos.y - offset_y
        img_x, img_y = self.clamp_to_image(img_x, img_y)
        click_point = wx.Point(img_x, img_y)

        # Check if click is inside any box
        for box in self.__boxes:
            if self.point_in_box(click_point, box):
                getLog().debug(f"Selected box {self._selected_box}")

                select_event = BoxSelectedEvent(self, box)
                wx.PostEvent(self, select_event)
                self.Refresh()
                self._selected_box = box
                return  # Do not start dragging

        if self._selected_box is not None:
            # Deselect the current box if clicking outside
            deselect_event = BoxDeselectedEvent(self)
            wx.PostEvent(self, deselect_event)
            self.Refresh()

        self._selected_box = None
        self.dragging = True
        self.start_pos = click_point

    def on_left_up(self, event: wx.MouseEvent) -> None:
        if self.dragging:
            mouse_pos: wx.Point = event.GetPosition()
            offset_x, offset_y = self.get_image_offset()
            img_x: int = mouse_pos.x - offset_x
            img_y: int = mouse_pos.y - offset_y
            img_x, img_y = self.clamp_to_image(img_x, img_y)
            self.end_pos = wx.Point(img_x, img_y)

            x1: int = self.start_pos.x
            y1: int = self.start_pos.y
            x2: int = self.end_pos.x
            y2: int = self.end_pos.y

            def to_img_coords(x: int, y: int) -> tuple[int, int]:
                bx: int = self.bmp_size[0]
                by: int = self.bmp_size[1]
                iw: int = self.img_size[0]
                ih: int = self.img_size[1]
                img_x: int = int(x / bx * iw)
                img_y: int = int(y / by * ih)
                return img_x, img_y

            img_start: tuple[int, int] = to_img_coords(x1, y1)
            img_end: tuple[int, int] = to_img_coords(x2, y2)
            coords: tuple[int, int, int, int] = (
                img_start[0], img_start[1],
                img_end[0] - img_start[0], img_end[1] - img_start[1]
            )
            self.undo_stack.append(('draw_box', copy.deepcopy(self.__boxes)))
            self.redo_stack.clear()
            self.add_new_box(coords, 'user')
            self.dragging = False

    def on_motion(self, event: wx.MouseEvent):
        if self.dragging and event.Dragging() and event.LeftIsDown():
            mouse_pos: wx.Point = event.GetPosition()
            offset_x, offset_y = self.get_image_offset()
            img_x: int = mouse_pos.x - offset_x
            img_y: int = mouse_pos.y - offset_y
            img_x, img_y = self.clamp_to_image(img_x, img_y)
            self.end_pos = wx.Point(img_x, img_y)
            self.Refresh()

    def _is_box_selected(self, box: BoxData) -> bool:
        if self._selected_box is None:
            return False
        """Check if the box is currently selected."""

        if self._selected_box == box or self._selected_box.coords == box.coords:
            return True

        return False

    @staticmethod
    def get_box_label_text(box: BoxData) -> str:
        """Return the label text for the box (to be implemented)."""
        label = ', '.join(box.tags)
        return label

    def on_paint(self, event: wx.PaintEvent):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        if self.bitmap:
            offset_x, offset_y = self.get_image_offset()
            dc.DrawBitmap(self.bitmap, offset_x, offset_y)
            # Swap image and bitmap dimensions for 90/270 degree rotations
            img_w, img_h = self.img_size
            bx, by = self.bmp_size
            if self.rotation_angle in (90, 270):
                img_w, img_h = img_h, img_w
                bx, by = by, bx
            for box in self.__boxes:
                # Check if the box is selected
                is_selected = self._selected_box is not None and self._is_box_selected(box)
                coords = box.coords  # (x, y, w, h) in original image space

                def rotate_point(x, y, w, h, angle, orig_w, orig_h):
                    if angle == 90:
                        return orig_h - y - h, x, h, w
                    elif angle == 180:
                        return orig_w - x - w, orig_h - y - h, w, h
                    elif angle == 270:
                        return y, orig_w - x - w, h, w
                    else:
                        return x, y, w, h

                # Use original image size for rotation
                x, y, w, h = rotate_point(coords[0], coords[1], coords[2], coords[3], self.rotation_angle,
                                          self.img_size[0], self.img_size[1])

                # Map image coordinates to bitmap coordinates using swapped dimensions
                bmp_x1 = int(x / img_w * bx)
                bmp_y1 = int(y / img_h * by)
                bmp_x2 = int((x + w) / img_w * bx)
                bmp_y2 = int((y + h) / img_h * by)

                rect = wx.Rect(bmp_x1 + offset_x, bmp_y1 + offset_y, bmp_x2 - bmp_x1, bmp_y2 - bmp_y1)
                stroke_width = 3 if is_selected else 1

                # Draw label area at the bottom edge
                label_text = self.get_box_label_text(box) or ""
                label_rect_height = 18  # px, adjust as needed
                label_rect = wx.Rect(
                    bmp_x1 + offset_x,
                    bmp_y2 + offset_y - label_rect_height,
                    bmp_x2 - bmp_x1,
                    label_rect_height
                )

                if label_text.startswith("unknown-"):
                    getLog().debug(f'{box} with unknown label.')

                colour: wx.Colour
                if box.source == 'user':
                    colour = wx.RED
                elif box.source == 'automatic':
                    colour = wx.BLUE
                else:
                    colour = wx.YELLOW
                dc.SetPen(wx.Pen(colour, stroke_width))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRectangle(rect)

                # Truncate text to fit box width
                dc.SetBrush(wx.Brush(colour))
                font = wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
                dc.SetFont(font)
                dc.SetTextForeground(wx.WHITE)
                max_width = label_rect.GetWidth() - 4
                truncated_text = label_text
                while dc.GetTextExtent(truncated_text)[0] > max_width and len(truncated_text) > 0:
                    truncated_text = truncated_text[:-1]
                if truncated_text != label_text and len(truncated_text) > 3:
                    truncated_text = truncated_text[:-3] + "..."

                dc.DrawRectangle(label_rect)
                dc.DrawText(truncated_text, label_rect.x + 2, label_rect.y + 2)

            # Draw current drag box
            if self.dragging and self.start_pos and self.end_pos:
                offset_x, offset_y = self.get_image_offset()
                rect = wx.Rect(
                    self.start_pos[0] + offset_x,
                    self.start_pos[1] + offset_y,
                    self.end_pos[0] - self.start_pos[0],
                    self.end_pos[1] - self.start_pos[1]
                )
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
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        if action[0] == 'draw_box':
            self.redo_stack.append(('draw_box', copy.deepcopy(self.__boxes)))
            self.__boxes = copy.deepcopy(action[1])
            # box_removed_event = BoxRemovedEvent(self, box)
            # wx.PostEvent(self, box_removed_event)
        elif action[0] == 'rotate':
            self.redo_stack.append(('rotate', self.rotation_angle, copy.deepcopy(self.__boxes)))
            self.rotation_angle = action[1]
            self.__boxes = copy.deepcopy(action[2])
        update_event = BoxUpdatedEvent(self, self.__boxes)
        wx.PostEvent(self, update_event)
        self.Refresh()

    def add_new_box(self, coords: tuple[int, int, int, int], source: str) -> None:
        """Add a new box with the given coordinates."""
        box_number = len(self.__boxes) + 1
        new_box_label = f"unknown-{box_number}"
        new_box = BoxData(coords, [new_box_label], 'user')
        self.__boxes.append(new_box)
        box_added_event = BoxAddedEvent(self, new_box)
        getLog().info(f'New box added: {new_box}, {box_added_event}')
        wx.PostEvent(self, box_added_event)
        self.Refresh()

    def redo(self):
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        if action[0] == 'draw_box':
            self.undo_stack.append(('draw_box', copy.deepcopy(self.__boxes)))
            self.__boxes = copy.deepcopy(action[1])
        elif action[0] == 'rotate':
            self.undo_stack.append(('rotate', self.rotation_angle, copy.deepcopy(self.__boxes)))
            self.rotation_angle = action[1]
            self.__boxes = copy.deepcopy(action[2])
        self.Refresh()

    def on_delete_box(self, box: BoxData) -> None:
        if box in self.__boxes:
            self.undo_stack.append(("delete_box", copy.deepcopy(self.__boxes)))
            self.__boxes.remove(box)
            self.Refresh()  # Redraw the image panel

    def on_add_tag(self, box: BoxData, tag_number: int, tag: str) -> None:
        if box in self.__boxes and tag not in box.tags:
            self.undo_stack.append(("add_tag", copy.deepcopy(self.__boxes)))
            box.tags.append(tag)
            self.Refresh()

    def on_remove_tag(self, box: BoxData, tag_number: int, tag: str) -> None:
        if box in self.__boxes and tag in box.tags:
            self.undo_stack.append(("remove_tag", copy.deepcopy(self.__boxes)))
            box.tags.remove(tag)
            self.Refresh()

    @property
    def boxes(self) -> list[BoxData]:
        """Get the list of boxes."""
        return self.__boxes

    @boxes.setter
    def boxes(self, new_boxes: list[BoxData]) -> None:
        """Set the list of boxes and clear undo/redo stacks."""
        self.__boxes = new_boxes
        self.Refresh()  # Redraw the image panel
