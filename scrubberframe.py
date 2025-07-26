import json
import os
from copy import copy
from typing import List, Dict

import cv2
import numpy as np
import wx

from boxdata import BoxData, Coordinate
from controlspanel import ControlsPanel
from events.BoxSelectedEvent import BoxSelectedEvent
from events.events import EVT_BOX_SELECTED
from imagepanel import ImagePanel
from logutil import getLog
from markerpanel import MarkerPanel  # Adjust import as needed
from tagpanel import TagPanel

def remove_empty(tags: List[str]) -> List[str]:
    """Remove empty tags from the list."""
    return [tag for tag in tags if tag.strip()]

def save_boxes_to_stream(stream, frame_boxes: dict[int, list[BoxData]]) -> None:
    # frame_boxes: {frame_number: [BoxData, ...]}
    serializable = {
        frame: [
            {"coords": box.coords, "tags": remove_empty(box.tags), "source": box.source}
            for box in boxes
        ]
        for frame, boxes in frame_boxes.items()
    }
    json.dump(serializable, stream)

def save_boxes_to_file(filename: str, frame_boxes: dict[int, list[BoxData]]) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        save_boxes_to_stream(f, frame_boxes)

def merge_duplicate_boxes(boxes: List[BoxData]) -> List[BoxData]:
    """Merge boxes with the same coordinates and tags."""
    merged: Dict[Coordinate, BoxData] = {}
    for box in boxes:
        key = (box.coords, tuple(sorted(box.tags)))
        if key not in merged:
            merged[key] = BoxData(coords=box.coords, tags=copy(box.tags), source=box.source)
        else:
            merged[key].tags.extend(box.tags)
            merged[key].tags = remove_empty(merged[key].tags)

    return list(merged.values())

def load_boxes_from_stream(stream) -> dict[int, list[BoxData]]:
    data = json.load(stream)
    return {
        int(frame): merge_duplicate_boxes([BoxData(tuple(box["coords"]), list(box["tags"]), box.get("source", "automatic")) for box in boxes])
        for frame, boxes in data.items()
    }

def load_boxes_from_file(filename: str) -> dict[int, list[BoxData]]:
    with open(filename, "r", encoding="utf-8") as f:
        return load_boxes_from_stream(f)


class ScrubberFrame(wx.Frame):
    __frame_boxes: Dict[int, List[BoxData]] = {}  # Map of frame index to BoxData
    # __boxes: List[BoxData] = []  # Or load from your data source
    __image_panel: ImagePanel
    __button_panel: ControlsPanel
    __box_data_filename: str | None = None

    @staticmethod
    def create_box_data_name_from_filename(file_name: str) -> str:
        """Create a box data name from the file name."""
        """Return the filename with its extension replaced by .json."""
        base, _ = os.path.splitext(file_name)
        return base + ".json"

    @property
    def box_data_filename(self) -> str | None:
        """Get the filename for saving/loading box data."""
        return self.__box_data_filename

    @box_data_filename.setter
    def box_data_filename(self, filename: str | None) -> None:
        """Set the filename for saving/loading box data."""
        if filename is not None and not filename.endswith('.json'):
            raise ValueError("Box data filename must end with .json")
        self.__box_data_filename = filename
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def load_box_data(self) -> Dict[int, List[BoxData]]:
        """Load box data from the specified file."""
        if self.box_data_filename and os.path.exists(self.box_data_filename):
            try:
                self.__frame_boxes = load_boxes_from_file(self.box_data_filename)
                count = self.count_boxes()
                getLog().info(f"Loaded {count} boxes in data from {self.box_data_filename}")
                return self.__frame_boxes
            except Exception as e:
                getLog().error(f"Error loading box data: {e}")
        else:
            getLog().warning("No box data file specified or file does not exist.")
        return {}

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
        )
        self.__image_panel.Bind(EVT_BOX_SELECTED, self.tag_panel.on_box_selected)
        frame_boxes = self.__get_frame_boxes(self._current_index)
        self.tag_panel.update_boxes(frame_boxes)
        # print(f'ScrubberFrame.__init__: TagPanel referencing {hex(id(self.__boxes))}=>{self.__boxes}')
        self.tag_panel.bind_box_events(self.__image_panel)
        # self.image_panel.Bind(EVT_BOX_ADDED, self.tag_panel.Refresh)

        image_and_tag_sizer.Add(self.tag_panel, 0, wx.EXPAND | wx.ALL, 10)

        vbox.Add(image_and_tag_sizer, 1, wx.EXPAND | wx.ALL, 5)

        # Add ControlsPanel below image_panel
        self.__button_panel = ControlsPanel(main_panel)
        vbox.Add(self.__button_panel, 0, wx.CENTER, 0)
        self.__button_panel.bind_buttons(self.on_prev, self.on_next, self.on_rotate_ccw, self.on_rotate_cw)

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

    def __get_frame_boxes(self, index: int) -> List[BoxData]:
        """Get the boxes for the current frame index."""
        fb = self.__frame_boxes
        if not index in fb:
            fb[index] = []

        return fb[index]

    def display_image(self):
        img = self.get_frame(self._current_index, self._rotation_angle)
        if img is None:
            return
        panel_size = self.__image_panel.GetSize()
        if panel_size.GetWidth() < 10 or panel_size.GetHeight() < 10:
            return  # Panel not yet sized, skip

        self.__image_panel.set_image(img, self._rotation_angle)
        self.__image_panel.boxes = self.__current_boxes

        frame_boxes = self.__get_frame_boxes(self._current_index)
        self.tag_panel.update_boxes(frame_boxes)

        self.__button_panel.set_prev_enabled(self._current_index > 0)
        self.__button_panel.set_next_enabled(self._current_index < self.num_frames)

        self.slider.SetValue(self._current_index)
        self.Refresh()

    def on_resize(self, event):
        self.display_image()
        event.Skip()

    def on_prev(self, event):
        if self._current_index > 0:
            self._current_index -= 1
            self.display_image()

    def frame_has_boxes(self, index: int) -> bool:
        """Check if the current frame has boxes."""
        return index in self.__frame_boxes and len(self.__frame_boxes[index]) > 0

    def on_next(self, event):
        if self._current_index < self.num_frames - 1:
            next_index = self._current_index + 1

            current_frame: np.ndarray | None = None
            next_frame: np.ndarray | None = None
            if not self.frame_has_boxes(next_index) and self._current_index in self.__frame_boxes:
                # locate boxes automatically
                current_frame = self.get_frame(self._current_index, self._rotation_angle)
                next_frame = self.get_frame(next_index, self._rotation_angle)

            frame_boxes = self.__current_boxes

            self._current_index += 1
            if next_frame is not None and current_frame is not None:
                found_boxes: list[BoxData] = []
                for box in frame_boxes:
                    new_bbox = self.find_object_in_next_frame(current_frame, next_frame, box)
                    if new_bbox is not None:
                        getLog().debug('Found new coordinates for box:', box, '->', new_bbox)
                        found_boxes.append(new_bbox)

                if next_index not in self.__frame_boxes:
                    self.__frame_boxes[next_index] = []
                self.__frame_boxes[next_index].extend(found_boxes)

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

    @property
    def __current_boxes(self) -> List[BoxData]:
        """Get the boxes for the current frame index."""
        return self.__get_frame_boxes(self._current_index)

    def on_box_update(self) -> None:
        """Called when a box is updated, e.g., after adding or removing a tag."""
        self.tag_panel.update_boxes(self.__current_boxes)
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

    def count_boxes(self):
        count = 0
        for box_list in self.__frame_boxes.values():
            count += len(box_list)
        """Count the number of boxes in the current frame."""
        return count

    def on_close(self, event):
        # Save boxes before exiting
        count = self.count_boxes()
        save_boxes_to_file(self.__box_data_filename, self.__frame_boxes)
        getLog().info(f'{count} boxes saved to {self.__box_data_filename}')
        event.Skip()  # Continue closing

    def on_box_selected(self, event: BoxSelectedEvent) -> None:
        selected_box = event.box
        getLog().debug(f'Selected box {selected_box.coords} with tags {selected_box.tags}')
        # self.tag_panel.set_selected(event.box)

    @staticmethod
    def find_object_in_next_frame(
        prev_frame: np.ndarray,
        next_frame: np.ndarray,
        bbox: BoxData
    ) -> BoxData | None:
        x: int
        y: int
        w: int
        h: int
        x, y, w, h = bbox.coords
        template: np.ndarray = prev_frame[y:y + h, x:x + w]

        orb: cv2.ORB = cv2.ORB_create()
        kp1: list[cv2.KeyPoint]
        des1: np.ndarray | None
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2: list[cv2.KeyPoint]
        des2: np.ndarray | None
        kp2, des2 = orb.detectAndCompute(next_frame, None)

        if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
            return None

        bf: cv2.BFMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches: list[cv2.DMatch] = bf.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        # src_pts: np.ndarray = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        # dst_pts: np.ndarray = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        src_pts = np.float32([np.array(kp1[m.queryIdx].pt) for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([np.array(kp2[m.trainIdx].pt) - np.array([x, y]) for m in matches]).reshape(-1, 1, 2)

        if src_pts.shape[0] >= 3:
            M: np.ndarray | None
            mask: np.ndarray | None
            M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            if M is not None:
                corners: np.ndarray = np.float32([
                    [x, y],
                    [x + w, y],
                    [x + w, y + h],
                    [x, y + h]
                ]).reshape(-1, 1, 2)
                new_corners: np.ndarray = cv2.transform(corners, M)
                new_bbox: tuple[int, int, int, int] = cv2.boundingRect(new_corners)

                new_data = BoxData(
                    coords=(new_bbox[0], new_bbox[1], new_bbox[2], new_bbox[3]),
                    tags=copy(bbox.tags),
                    source='automatic'
                )
                return new_data
        return None