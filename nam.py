import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import io
from PIL import Image

import torchvision.transforms as transforms
import numpy as np
import tensorflow as tf
import torch
import torchvision.models as models
import qimage2ndarray
#from pytorchmnist import CNN
class MyApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.k = -1
        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 30
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.loaded_model = None
        self.initUI()

    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu('File')

        load_model_action = QAction('Load model', self)
        load_model_action.setShortcut('Ctrl+L')
        load_model_action.triggered.connect(self.load_model)

        save_action = QAction('Save', self)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save)

        clear_action = QAction('Clear', self)
        clear_action.setShortcut('Ctrl+C')
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(load_model_action)
        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.statusbar = self.statusBar()

        self.setWindowTitle('MNIST Classifier')
        self.setGeometry(300, 300, 400, 400)
        self.show()

    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False
            if self.k == 1:
                arr = np.zeros((28, 28))
                for i in range(28):
                    for j in range(28):
                        arr[j, i] = 1 - self.image.scaled(28, 28).pixelColor(i, j).getRgb()[0] / 255.0
                arr = arr.reshape(-1, 28, 28)

                if self.loaded_model:
                    pred = self.loaded_model.predict(arr)[0]
                    pred_num = str(np.argmax(pred))
                    self.statusbar.showMessage('예측값은 ' + pred_num + '입니다.')
            elif self.k == 0:
                arr = np.zeros((28, 28))
                for i in range(28):
                    for j in range(28):
                        arr[j, i] = 1 - self.image.scaled(28, 28).pixelColor(i, j).getRgb()[0] / 255.0
                arr = arr.reshape(-1, 28, 28)
                arr = torch.Tensor(arr)
                arr = arr.view(-1,28*28)

                with torch.no_grad():
                    self.loaded_model.eval()
                    output = self.loaded_model(arr)
                    index = output.data.cpu().numpy().argmax()
                    pred_num = str(index)
                    self.statusbar.showMessage('예측값은 ' + pred_num + '입니다.')



    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Load Model', '')
        print(fname)
        print(type(fname))
        ff = fname.split('.')
        print(ff[1])
        if ff[1]=='h5':
            self.k = 1
            if fname:
                self.loaded_model = tf.keras.models.load_model(fname)
                self.statusbar.showMessage('tensorflow Model loaded.')
        elif ff[1]=='pt':
            self.k = 0
            if fname:
                self.loaded_model = torch.jit.load(fname)
                self.loaded_model.eval()
                self.statusbar.showMessage('pytorch Model loaded.')

    def save(self):
        fpath, _ = QFileDialog.getSaveFileName(self, 'Save Image', '', "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if fpath:
            self.image.scaled(28, 28).save(fpath)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        self.statusbar.clearMessage()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    sys.exit(app.exec_())