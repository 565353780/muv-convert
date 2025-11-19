'''
import OCC.Display.SimpleGui
# OCC.Display.SimpleGui.init_display.centerOnScreen
# You need to add int() for centerOnScreen func first
def centerOnScreen(self) -> None:
    resolution = QtWidgets.QApplication.desktop().screenGeometry()
    x = int((resolution.width() - self.frameSize().width()) / 2)
    y = int((resolution.height() - self.frameSize().height()) / 2)
    self.move(x, y)
'''

from occwl.solid import Solid
from occwl.viewer import Viewer

box = Solid.make_box(10, 10, 10)
v = Viewer()
v.display(box)
v.fit()
v.show()
