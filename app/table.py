
import sys
import numpy as np
import matplotlib.pyplot as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt6 import QtGui, QtCore
from PyQt6.QtWidgets import QMainWindow, QTableWidget, QTableWidgetItem, QHeaderView
from PyQt6 import QtWidgets
from PyQt6 import QtGui, QtCore

def mathTex_to_QPixmap(mathTex, fs):

    #---- set up a mpl figure instance ----

    mpl.rcParams['text.usetex'] = False

    fig = mpl.figure()
    fig.patch.set_facecolor('none')
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    #---- plot the mathTex expression ----

    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.patch.set_facecolor('none')
    t = ax.text(0, 0, mathTex, ha='left', va='bottom', fontsize=fs)

    #---- fit figure size to text artist ----

    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)

    text_bbox = t.get_window_extent(renderer)

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight)
    # set fig size
    

    #---- convert mpl figure to QPixmap ----

    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(
        QtGui.QImage(buf, 
                     size[0], 
                     size[1],
                     QtGui.QImage.Format.Format_ARGB32))
    qpixmap = QtGui.QPixmap(qimage)

    return qpixmap

class TexQTableWidget(QTableWidget):   
    def __init__(self, parent=None):
        super(TexQTableWidget, self).__init__(parent)

        self.setHorizontalHeader(TexHorizHeader(self))
        self.setVerticalHeader(TexVertHeader(self))

    def setHorizontalHeaderLabels(self, headerLabels, fontsize = 12):

        qpixmaps = []
        indx = 0
        for labels in headerLabels:
            qpixmaps.append(mathTex_to_QPixmap(labels, fontsize))            
            self.setColumnWidth(indx, qpixmaps[indx].size().width() + 16)
            indx += 1

        self.horizontalHeader().qpixmaps = qpixmaps

        super(TexQTableWidget, self).setHorizontalHeaderLabels(headerLabels)
    
    def setVerticalHeaderLabels(self, rowLabels, fontsize = 12):

        qpixmaps = []
        widths = []
        indx = 0
        for labels in rowLabels:
            qpixmaps.append(mathTex_to_QPixmap(labels, fontsize))
            widths.append(qpixmaps[indx].size().width() + 16)
            indx += 1

        self.verticalHeader().setMinimumWidth(max(widths))
        self.verticalHeader().qpixmaps = qpixmaps
        super(TexQTableWidget, self).setVerticalHeaderLabels(rowLabels)

class TexHorizHeader(QHeaderView):
    def __init__(self, parent):
        super(TexHorizHeader, self).__init__(QtCore.Qt.Orientation.Horizontal, parent)

        self.setSectionsClickable(True)
        self.setStretchLastSection(True)

        self.qpixmaps = []

    def paintSection(self, painter, rect, logicalIndex):

        if not rect.isValid():
            return
        
        if not painter.isActive():
            return

        #------------------------------ paint section (without the label) ----

        opt = QtWidgets.QStyleOptionHeader()

        self.initStyleOption(opt)

        opt.rect = rect
        opt.section = logicalIndex
        opt.text = ""

        #---- mouse over highlight ----

        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())               
        if rect.contains(mouse_pos):
            opt.state |= QtWidgets.QStyle.StateFlag.State_MouseOver

        #---- paint ----

        painter.save()        
        self.style().drawControl(QtWidgets.QStyle.ControlElement.CE_Header, opt, painter, self)
        painter.restore()

        #------------------------------------------- paint mathText label ----

        qpixmap = self.qpixmaps[logicalIndex]

        #---- centering ----

        xpix = (rect.width() - qpixmap.size().width()) / 2. + rect.x()
        ypix = (rect.height() - qpixmap.size().height()) / 2.

        #---- paint ----
        aleft, atop = int(xpix), int(ypix)
        awidth, aheight = qpixmap.size().width(), qpixmap.size().height()
        rect = QtCore.QRect(aleft, atop, awidth, aheight)
        painter.drawPixmap(rect, qpixmap)        

    def sizeHint(self):

        baseSize = super(TexHorizHeader, self).sizeHint()

        baseHeight = baseSize.height()
        if len(self.qpixmaps):
            for pixmap in self.qpixmaps:
               baseHeight = max(pixmap.height() + 8, baseHeight)
        baseSize.setHeight(baseHeight)

        return baseSize
    
class TexVertHeader(QHeaderView):

    def __init__(self, parent):
        super(TexVertHeader, self).__init__(QtCore.Qt.Orientation.Vertical, parent)

        self.setSectionsClickable(True)
        self.qpixmaps = []
    
    def paintSection(self, painter, rect, logicalIndex):

        if not rect.isValid():
            return
        
        if not painter.isActive():
            return

        #------------------------------ paint section (without the label) ----

        opt = QtWidgets.QStyleOptionHeader()

        self.initStyleOption(opt)

        opt.rect = rect
        opt.section = logicalIndex
        opt.text = ""

        #---- mouse over highlight ----

        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())               
        if rect.contains(mouse_pos):
            opt.state |= QtWidgets.QStyle.StateFlag.State_MouseOver

        #------------------------------------------- paint mathText label ----
        try:
            qpixmap = self.qpixmaps[logicalIndex]
        except IndexError:
            return

        painter.save()        
        self.style().drawControl(QtWidgets.QStyle.ControlElement.CE_Header, opt, painter, self)
        painter.restore()

        xpix = (rect.width() - qpixmap.size().width()) / 2.
        ypix = (rect.height() - qpixmap.size().height()) / 2. + rect.y()

        #---- paint ----
        aleft, atop = int(xpix), int(ypix)
        awidth, aheight = qpixmap.size().width(), qpixmap.size().height()
        rect = QtCore.QRect(aleft, atop, awidth, aheight)
        painter.drawPixmap(rect, qpixmap)
    
    def sizeHint(self):

        baseSize = super(TexVertHeader, self).sizeHint()

        baseHeight = baseSize.height()
        if len(self.qpixmaps):
            for pixmap in self.qpixmaps:
               baseHeight = max(pixmap.height() + 8, baseHeight)
        baseSize.setHeight(baseHeight)

        return baseSize




