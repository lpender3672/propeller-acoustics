import sys

import matplotlib.pyplot as mpl
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import (
    QComboBox,
    QHeaderView,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
)


def mathTex_to_QPixmap(mathTex, fs, font_colour=None):

    # ---- set up a mpl figure instance ----
    mpl.rcParams["text.usetex"] = False

    # Set the DPI for higher resolution rendering
    fig = mpl.figure()
    fig.patch.set_facecolor("none")
    fig.set_canvas(FigureCanvasAgg(fig))
    renderer = fig.canvas.get_renderer()

    # ---- plot the mathTex expression ----
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.patch.set_facecolor("none")
    t = ax.text(0, 0, mathTex, ha="left", va="bottom", fontsize=fs, color=font_colour)

    # ---- fit figure size to text artist ----
    fwidth, fheight = fig.get_size_inches()
    fig_bbox = fig.get_window_extent(renderer)
    text_bbox = t.get_window_extent(renderer)

    tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
    tight_fheight = text_bbox.height * fheight / fig_bbox.height

    fig.set_size_inches(tight_fwidth, tight_fheight)

    # ---- convert mpl figure to QPixmap ----
    buf, size = fig.canvas.print_to_buffer()
    qimage = QtGui.QImage.rgbSwapped(
        QtGui.QImage(buf, size[0], size[1], QtGui.QImage.Format.Format_ARGB32)
    )

    qpixmap = QtGui.QPixmap(qimage)
    # scaled_size = qpixmap.size() * 100 / dpi
    # qpixmap = qpixmap.scaled(scaled_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)

    return qpixmap


class TexQTableWidget(QTableWidget):
    def __init__(self, parent=None, font_colour=None):
        super(TexQTableWidget, self).__init__(parent)

        font_colour = self.palette().color(QtGui.QPalette.ColorRole.Text)
        # convert QColor to hex matplotlib style
        hexcolour = "#{:02x}{:02x}{:02x}".format(
            font_colour.red(), font_colour.green(), font_colour.blue()
        )
        self.font_hexcolour = hexcolour

        self.setHorizontalHeader(TexHorizHeader(self))
        self.setVerticalHeader(TexVertHeader(self))

    def setHorizontalHeaderLabels(self, headerLabels, fontsize=12):

        qpixmaps = []
        indx = 0
        for labels in headerLabels:
            qpixmaps.append(mathTex_to_QPixmap(labels, fontsize, self.font_hexcolour))
            self.setColumnWidth(indx, qpixmaps[indx].size().width() + 16)
            indx += 1

        self.horizontalHeader().qpixmaps = qpixmaps

        super(TexQTableWidget, self).setHorizontalHeaderLabels(headerLabels)

    def setVerticalHeaderLabels(self, rowLabels, fontsize=12):

        qpixmaps = []
        widths = []
        indx = 0
        for labels in rowLabels:
            qpixmaps.append(mathTex_to_QPixmap(labels, fontsize, self.font_hexcolour))
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

        # ------------------------------ paint section (without the label) ----

        opt = QtWidgets.QStyleOptionHeader()

        self.initStyleOption(opt)

        opt.rect = rect
        opt.section = logicalIndex
        opt.text = ""

        # ---- mouse over highlight ----

        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        if rect.contains(mouse_pos):
            opt.state |= QtWidgets.QStyle.StateFlag.State_MouseOver

        # ---- paint ----
        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_Header, opt, painter, self
        )
        painter.restore()

        # ------------------------------------------- paint mathText label ----

        qpixmap = self.qpixmaps[logicalIndex]

        # ---- centering ----

        xpix = (rect.width() - qpixmap.size().width()) / 2.0 + rect.x()
        ypix = (rect.height() - qpixmap.size().height()) / 2.0

        # ---- paint ----
        aleft, atop = int(xpix), int(ypix)
        awidth, aheight = qpixmap.size().width(), qpixmap.size().height()
        rect = QtCore.QRect(aleft, atop, awidth, aheight)
        painter.drawPixmap(rect, qpixmap)

    def sizeHint(self):

        baseSize = super(TexHorizHeader, self).sizeHint()

        baseWidth = baseSize.width()
        if len(self.qpixmaps):
            for pixmap in self.qpixmaps:
                baseWidth = max(pixmap.width() + 8, baseWidth)
        baseSize.setWidth(baseWidth)

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

        # ------------------------------ paint section (without the label) ----

        opt = QtWidgets.QStyleOptionHeader()

        self.initStyleOption(opt)

        opt.rect = rect
        opt.section = logicalIndex
        opt.text = ""

        # ---- mouse over highlight ----

        mouse_pos = self.mapFromGlobal(QtGui.QCursor.pos())
        if rect.contains(mouse_pos):
            opt.state |= QtWidgets.QStyle.StateFlag.State_MouseOver

        # ------------------------------------------- paint mathText label ----
        try:
            qpixmap = self.qpixmaps[logicalIndex]
        except IndexError:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing)
        self.style().drawControl(
            QtWidgets.QStyle.ControlElement.CE_Header, opt, painter, self
        )
        painter.restore()

        xpix = (rect.width() - qpixmap.size().width()) / 2.0
        ypix = (rect.height() - qpixmap.size().height()) / 2.0 + rect.y()

        # ---- paint ----
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


class TableVar:
    def __init__(self, symbol, unit, description, dtype=float, sf=12):
        self.symbol = symbol
        self.unit = unit
        self.description = description
        self.dtype = dtype
        self.value = None
        self.sf = sf


class TableBox(QComboBox):
    def __init__(self, items, symbol, description):
        super().__init__()
        self.addItems(items)

        self.symbol = symbol
        self.description = description
        self.unit = ""
        self.dtype = str
        self.value = items[0].lower()


class OutputTable(TexQTableWidget):
    def __init__(self, vars, parent):
        super().__init__(parent)

        self.vars = vars

        self.setColumnCount(3)
        self.setRowCount(len(self.vars))

        self.setHorizontalHeaderLabels(["Value", "Units", "Description"], 12)

        self.assemble_table()

    def assemble_table(self):
        row_names = []
        for i, v in enumerate(self.vars):
            row_names.append(v.symbol)
            if isinstance(v, TableVar):
                self.setItem(i, 0, QTableWidgetItem())
            elif isinstance(v, TableBox):
                self.setCellWidget(i, 0, v)
            self.setItem(i, 1, QTableWidgetItem(v.unit))
            self.setItem(i, 2, QTableWidgetItem(v.description))
        if len(row_names) > 0:
            self.setVerticalHeaderLabels(row_names, 16)

        self.resizeColumnsToContents()

    def _round_sigfigs(self, x, sf):
        # scottgigante on stackoverflow
        x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (sf - 1))
        mags = 10 ** (sf - 1 - np.floor(np.log10(x_positive)))
        return np.round(x * mags) / mags

    def set_values(self):

        for i, v in enumerate(self.vars):
            if isinstance(v, TableVar):
                val = v.value
                if v.dtype == float and val is not None:
                    val = self._round_sigfigs(val, v.sf)
                self.item(i, 0).setText(str(val))
            elif isinstance(v, TableBox):
                # set index of combobox
                self.cellWidget(i, 0).setCurrentText(v.value)

        self.resizeColumnsToContents()


class InputTable(OutputTable):

    def __init__(self, vars, parent):
        super().__init__(vars, parent)

        for i in range(self.rowCount()):
            v = self.cellWidget(i, 0)
            if isinstance(v, TableBox):
                v.currentIndexChanged.connect(
                    lambda index, r=i, c=0: self.cellChanged.emit(r, c)
                )
        self.cellChanged.connect(self.on_cell_changed)

    def on_cell_changed(self, row, col):  # index is for comboboxes
        if col != 0:
            return

        try:
            v = self.vars[row]
        except IndexError:
            return

        # disable signal

        self.blockSignals(True)  # not best way but reliable for inherited class

        if isinstance(v, TableBox):
            v.value = v.currentText()

        elif isinstance(v, TableVar):
            try:
                v.dtype(self.item(row, col).text())
            except ValueError:
                self.item(row, col).setBackground(QtGui.QColor(255, 0, 0))
                self.clearSelection()
            except AttributeError:
                print(self.item(row, col))
            else:
                v.value = v.dtype(self.item(row, col).text())
                # get color of theme
                colour = self.palette().color(QtGui.QPalette.ColorRole.Base)
                self.item(row, col).setBackground(colour)
                self.item(row, col).setText(str(v.value))

        self.blockSignals(False)

    def set_values(self):

        self.blockSignals(True)
        super().set_values()
        self.blockSignals(False)
