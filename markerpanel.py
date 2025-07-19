import wx


class MarkerPanel(wx.Panel):
    def __init__(self, parent, num_images, interval=5):
        super().__init__(parent, size=wx.Size(-1, 30))
        self.num_images = num_images
        self.interval = interval
        self.Bind(wx.EVT_PAINT, self.on_paint)

    def on_paint(self, event):
        dc = wx.PaintDC(self)
        width = self.GetSize().GetWidth()
        last_x = -32  # Start so first marker is always drawn
        for i in range(0, self.num_images, self.interval):
            x = int(i * width / max(1, self.num_images - 1))
            if x - last_x >= 32:
                dc.DrawLine(x, 0, x, 15)
                dc.DrawText(str(i), x - 10, 16)
                last_x = x
