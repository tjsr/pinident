import wx

wxEVT_BOX_ADDED = wx.NewEventType()
wxEVT_BOX_REMOVED = wx.NewEventType()
wxEVT_BOX_UPDATED = wx.NewEventType()
wxEVT_BOX_EDITED = wx.NewEventType()
wxEVT_BOX_LABEL_EDITED = wx.NewEventType()

# idEVT_BOX_ADDED = wx.NewId()
# idEVT_BOX_REMOVED = wx.NewId()
# idEVT_BOX_UPDATED = wx.NewId()
# idEVT_BOX_EDITED = wx.NewId()
# idEVT_BOX_LABEL_EDITED = wx.NewId()

# peEVT_BOX_LABEL_EDITED = wx.PyEventBinder(wx.NewId())
# peEVT_BOX_EDITED = wx.PyEventBinder(wx.NewId())
# peEVT_BOX_UPDATED = wx.PyEventBinder(wx.NewId())
# peEVT_BOX_REMOVED = wx.PyEventBinder(wx.NewId())

EVT_BOX_ADDED = wx.PyEventBinder(wxEVT_BOX_ADDED, 1)
EVT_BOX_REMOVED = wx.PyEventBinder(wxEVT_BOX_REMOVED, 1)
EVT_BOX_UPDATED = wx.PyEventBinder(wxEVT_BOX_UPDATED, 1)
EVT_BOX_EDITED = wx.PyEventBinder(wxEVT_BOX_EDITED, 1)
EVT_BOX_LABEL_EDITED = wx.PyEventBinder(wxEVT_BOX_LABEL_EDITED, 1)