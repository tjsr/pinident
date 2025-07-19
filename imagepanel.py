import numpy as np
import wx
import copy

from boxdata import BoxData
from events.BoxAddedEvent import BoxAddedEvent
from events.BoxUpdatedEvent import BoxUpdatedEvent

RotationAngle = int
ImageSize = tuple[int, int] # (width, height)
UserAction = str  # 'draw_box', 'rotate', etc.

class ImagePanel(wx.Panel, wx.PyEventBinder):
    image: np.ndarray | None
    bitmap: wx.Bitmap | None
    __boxes: list[BoxData]
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
        # Save current state for undo
        self.undo_stack.append(('rotate', self.rotation_angle, copy.deepcopy(self.__boxes)))
        self.redo_stack.clear()

        """Rotate all boxes to match the new image rotation."""
        angle_diff = (new_angle - self.rotation_angle) % 360
        w, h = self.img_size
        def rotate_point(x: int, y: int) -> tuple[int, int]:
            if angle_diff == 90:
                return h - y - 1, x
            elif angle_diff == 180:
                return w - x - 1, h - y - 1
            elif angle_diff == 270:
                return y, w - x - 1
            else:
                return x, y

        new_boxes: list[BoxData] = []
        for box in self.__boxes:
            coords = box.coords
            x1, y1 = coords[0], coords[1]
            x2, y2 = coords[0] + coords[2], coords[1] + coords[3]
            new_start = rotate_point(x1, y1)
            new_end = rotate_point(x2, y2)
            new_coords = (new_start[0], new_start[1], new_end[0] - new_start[0], new_end[1] - new_start[1])
            new_boxes.append(BoxData(new_coords, box.tags.copy()))

        self.__boxes = new_boxes
        self.rotation_angle = new_angle
        update_event = BoxUpdatedEvent(self, self.__boxes)
        wx.PostEvent(self, update_event)
        self.Refresh()

    def get_image_offset(self) -> tuple[int, int]:
        """Return the (x, y) offset of the image inside the panel."""
        panel_w: int = self.GetSize().GetWidth()
        panel_h: int = self.GetSize().GetHeight()
        img_w: int = self.bmp_size[0]
        img_h: int = self.bmp_size[1]
        offset_x: int = (panel_w - img_w) // 2
        offset_y: int = (panel_h - img_h) // 2
        return offset_x, offset_y

    def clamp_to_image(self, x: int, y: int) -> tuple[int, int]:
        """Clamp coordinates to the image area."""
        img_w: int = self.bmp_size[0]
        img_h: int = self.bmp_size[1]
        x_clamped: int = max(0, min(x, img_w - 1))
        y_clamped: int = max(0, min(y, img_h - 1))
        return x_clamped, y_clamped

    def on_left_down(self, event: wx.MouseEvent) -> None:
        self.dragging = True
        mouse_pos: wx.Point = event.GetPosition()
        offset_x, offset_y = self.get_image_offset()
        img_x: int = mouse_pos.x - offset_x
        img_y: int = mouse_pos.y - offset_y
        img_x, img_y = self.clamp_to_image(img_x, img_y)
        self.start_pos = wx.Point(img_x, img_y)

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
            self.add_new_box(coords)
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

    def on_paint(self, event: wx.PaintEvent):
        dc = wx.BufferedPaintDC(self)
        dc.Clear()
        if self.bitmap:
            dc.DrawBitmap(self.bitmap, (self.GetSize().GetWidth() - self.bmp_size[0]) // 2,
                          (self.GetSize().GetHeight() - self.bmp_size[1]) // 2)
            # Draw boxes
            for box in self.__boxes:
                coords = box.coords
                bx, by = self.bmp_size
                iw, ih = self.img_size

                def to_bmp_coords(ix, iy):
                    bmp_x = int(ix / iw * bx)
                    bmp_y = int(iy / ih * by)
                    return bmp_x, bmp_y

                bmp_start = to_bmp_coords(coords[0], coords[1])
                bmp_end = to_bmp_coords(coords[0] + coords[2], coords[1] + coords[3])
                offset_x = (self.GetSize().GetWidth() - bx) // 2
                offset_y = (self.GetSize().GetHeight() - by) // 2
                rect = wx.Rect(bmp_start[0] + offset_x, bmp_start[1] + offset_y,
                               bmp_end[0] - bmp_start[0], bmp_end[1] - bmp_start[1])
                dc.SetPen(wx.Pen(wx.RED, 2))
                dc.SetBrush(wx.TRANSPARENT_BRUSH)
                dc.DrawRectangle(rect)
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

    def add_new_box(self, coords: tuple[int, int, int, int]) -> None:
        """Add a new box with the given coordinates."""
        new_box = BoxData(coords, [])
        self.__boxes.append(new_box)
        box_added_event = BoxAddedEvent(self, new_box)
        print("ImagePanel.add_new_box: New box added:", new_box, box_added_event)
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