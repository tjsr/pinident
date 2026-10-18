import wx


def marker_labels(num_images, width, measure, interval=5, gap=8):
    """Return (frame, tick x, text left, text width) for readable frame marks."""
    if num_images <= 0 or width <= 0:
        return []

    last_frame = num_images - 1
    text_width = lambda frame: measure(str(frame))[0]
    if text_width(0) > width:
        return []

    def placement(frame):
        tick_x = round(frame * (width - 1) / max(1, last_frame))
        label_width = text_width(frame)
        left = max(0, min(tick_x - label_width // 2, width - label_width))
        return frame, tick_x, left, label_width

    first = placement(0)
    if last_frame == 0:
        return [first]

    last = placement(last_frame) if text_width(last_frame) <= width else None
    max_label_width = max(first[3], last[3] if last else 0)
    required_step = (max_label_width + gap) * last_frame // max(1, width - 1) + 1
    base = max(1, interval)
    magnitude = 1
    while True:
        steps = (base * magnitude, 2 * base * magnitude, 5 * base * magnitude)
        step = next((value for value in steps if value >= required_step), None)
        if step is not None:
            break
        magnitude *= 10

    labels = [first]
    previous_right = first[2] + first[3]
    last_left = last[2] if last else width + gap
    for frame in range(step, last_frame, step):
        mark = placement(frame)
        if (mark[2] >= previous_right + gap and
                mark[2] + mark[3] + gap <= last_left):
            labels.append(mark)
            previous_right = mark[2] + mark[3]
    if last and last[2] >= previous_right + gap:
        labels.append(last)
    return labels


class MarkerPanel(wx.Panel):
    def __init__(self, parent, num_images, interval=5):
        super().__init__(parent, size=wx.Size(-1, 30))
        self.num_images = num_images
        self.interval = interval
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def on_size(self, event: wx.SizeEvent):
        self.Refresh()
        event.Skip()

    def on_paint(self, event: wx.PaintEvent):
        dc = wx.PaintDC(self)
        width = self.GetClientSize().width
        for frame, tick_x, text_left, _text_width in marker_labels(
                self.num_images, width, dc.GetTextExtent, self.interval):
            dc.DrawLine(tick_x, 0, tick_x, 15)
            dc.DrawText(str(frame), text_left, 16)
