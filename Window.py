from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QMenuBar, QAction, QFileDialog, QVBoxLayout,QPushButton, QLabel,QDockWidget,QLineEdit,QWidget
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QPoint
import sys
import numpy as np
import math
import h5py
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
class Window(QMainWindow):
    def __init__(self):
        super().__init__()


        title = "Number Detector"
        top = 784
        left = 784
        width = 784
        height = 784

        icon = "project_nn.png"

        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)
        self.setWindowIcon(QIcon(icon))

        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(QColor(0,0,0,127))
        self.drawing = False
        self.brushSize = 28
        self.brushColor = Qt.white
        self.lastPoint = QPoint()
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        brushSize = mainMenu.addMenu("Brush Size")
        brushColor = mainMenu.addMenu("Brush Color")

        saveAction = QAction(QIcon("icons/save.png"), "Save",self)
        saveAction.setShortcut("Ctrl+S")
        fileMenu.addAction(saveAction)
        saveAction.triggered.connect(self.save)

        predictAction = QAction(QIcon("icons/predict.png"), "Predict",self)
        predictAction.setShortcut("Ctrl+P")
        fileMenu.addAction(predictAction)
        predictAction.triggered.connect(self.predict)

        clearAction = QAction(QIcon("icons/clear.png"), "Clear", self)
        clearAction.setShortcut("Ctrl+C")
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        threepxAction = QAction( QIcon("icons/threepx.png"), "BasePixel", self)
        brushSize.addAction(threepxAction)
        threepxAction.triggered.connect(self.basePixel)

        fivepxAction = QAction(QIcon("icons/fivepx.png"), "SuperPixel", self)
        brushSize.addAction(fivepxAction)
        fivepxAction.triggered.connect(self.superPixel)

        sevenpxAction = QAction(QIcon("icons/sevenpx.png"),"MegaPixel", self)
        brushSize.addAction(sevenpxAction)
        sevenpxAction.triggered.connect(self.megaPixel)

        ninepxAction = QAction(QIcon("icons/ninepx.png"), "UltraPixel", self)
        brushSize.addAction(ninepxAction)
        ninepxAction.triggered.connect(self.ultraPixel)

        blackAction = QAction(QIcon("icons/black.png"), "Black", self)
        blackAction.setShortcut("Ctrl+B")
        brushColor.addAction(blackAction)
        blackAction.triggered.connect(self.blackColor)


        whitekAction = QAction(QIcon("icons/white.png"), "White", self)
        whitekAction.setShortcut("Ctrl+W")
        brushColor.addAction(whitekAction)
        whitekAction.triggered.connect(self.whiteColor)


        redAction = QAction(QIcon("icons/red.png"), "Red", self)
        redAction.setShortcut("Ctrl+R")
        brushColor.addAction(redAction)
        redAction.triggered.connect(self.redColor)

        greenAction = QAction(QIcon("icons/green.png"), "Green", self)
        greenAction.setShortcut("Ctrl+G")
        brushColor.addAction(greenAction)
        greenAction.triggered.connect(self.greenColor)

        yellowAction = QAction(QIcon("icons/yellow.png"), "Yellow", self)
        yellowAction.setShortcut("Ctrl+Y")
        brushColor.addAction(yellowAction)
        yellowAction.triggered.connect(self.yellowColor)

        

        

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()
            #print(self.lastPoint)


    def mouseMoveEvent(self, event):
        if(event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()



    def mouseReleaseEvent(self, event):

        if event.button() == Qt.LeftButton:
            self.drawing = False


    def paintEvent(self, event):
        canvasPainter  = QPainter(self)
        canvasPainter.drawImage(self.rect(),self.image, self.image.rect() )

    def myfunc(self,x):
        return 1 / (1 + math.exp(-x))

    def save(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if filePath == "":
            return
        self.image.save(filePath)

    def predict(self):
        incomingImage = self.image
        width = incomingImage.width()
        height = incomingImage.height()
        ptr = incomingImage.bits()
        ptr.setsize(height * width * 4)
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width,4))
        arr1 = arr[:,:,1]
        arr2 = arr1.reshape((28,arr1.shape[0]//28,28,-1)).mean(axis=3).mean(1)
        arr2 = arr2.reshape(1, 28, 28, 1)
        arr2 = (arr2)/255
        reconstructed_model = load_model("final_model.h5")
        result = reconstructed_model.predict_classes(arr2)
        print('Number = ',result[0])
        self.update()

    def clear(self):
        self.image.fill(QColor(0,0,0,127))
        self.update()


    def basePixel(self):
        self.brushSize = 7

    def superPixel(self):
        self.brushSize = 14

    def megaPixel(self):
        self.brushSize = 32

    def ultraPixel(self):
        self.brushSize = 56


    def blackColor(self):
        self.brushColor = Qt.black

    def whiteColor(self):
        self.brushColor = Qt.white

    def redColor(self):
        self.brushColor = Qt.red

    def greenColor(self):
        self.brushColor = Qt.green

    def yellowColor(self):
        self.brushColor = Qt.yellow




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()