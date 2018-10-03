from __future__ import print_function
import matplotlib.pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np
import sys

class stkshow(object):
    def __init__(self, X, External = False, min_thresh=5,max_thresh=95):
        if External==False:
            self.fig, self.ax = plt.subplots(1,1)
            self.ax.set_title('Image Stack') # Title
            self.X = X
            rows, cols, self.slices = X.shape
            self.ind = self.slices//2
            self.vmin = np.percentile(self.X,min_thresh)
            self.vmax = np.percentile(self.X,max_thresh)
            self.im = self.ax.imshow(self.X[:, :, self.ind], vmin=self.vmin, vmax=self.vmax)
            self.fig.canvas.mpl_connect('scroll_event', self.onscroll)
            self.update()
        else:
            self.stkshow_window(X)

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()
        plt.show()
        
def stkshowgui(data):
    # determine if you need to start a Qt app. 
    # If running from Spyder, the answer is a no.
    # From cmd, yes. From nb, not sure actually.
    if not QtGui.QApplication.instance():
        app = QtGui.QApplication([])
    else:
        app = QtGui.QApplication.instance()

    ## Create window with ImageView widget
    win = QtGui.QMainWindow()
    win.resize(800,800)
    imv = pg.ImageView()
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('Fetching image stack...')




    resizeflg = 0;
    maxxysize = 800;
    maxdataxySc = np.floor(max(data.shape[0],data.shape[1])/maxxysize).astype('int')
    if maxdataxySc>1:
        resizeflg = 1;


    from numba import jit

    @jit(nopython = True)
    def DownScale(imgin): #use 2x downscaling for scrol speed   
       # imgout = downscale_local_mean(imgin,(Sc, Sc)
        imgout = (imgin[0::2,0::2]+imgin[1::2,0::2]+imgin[0::2,1::2]+imgin[1::2,1::2])/4
        return imgout




    if len(data.shape)==4:#RGB assume xytc
        if data.shape[3]==3 or data.shape[3]==4:
            if resizeflg:
                dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc,1,1]))).astype('int'),dtype = 'uint16')    
                for i in range(0,data.shape[2]):
                    for j in range(0,data.shape[3]):
                        dataRs[:,:,i,j] = DownScale(data[:,:,i,j])
                dataRs = dataRs.transpose((2,0,1,3))
            else:
                dataRs = data;
                dataRs = dataRs.transpose((2,0,1,3))
        else:
            sys.exit('color channel needs to be RGB or RGBA')
    elif len(data.shape)==3:
        if resizeflg:
            dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc,1]))).astype('int'),dtype = 'uint16')    
            for i in range(0,data.shape[2]):
                dataRs[:,:,i] = DownScale(data[:,:,i])
            dataRs = dataRs.transpose([2,0,1])
        else:
            dataRs = data;
            dataRs = dataRs.transpose([2,0,1])

    elif len(data.shape)==2:
        if resizeflg:
            dataRs = np.zeros((np.ceil(data.shape/np.array([maxdataxySc,maxdataxySc]))).astype('int'),dtype = 'uint16')
            dataRs = DownScale(data)
        else:
            dataRs = data;
    else:
        print('Data must be 2D image or 3D image stack')



    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')


    win.setWindowTitle('Stack')

    ## Display the data and assign each frame a 
    imv.setImage(dataRs)#, xvals=np.linspace(1., dataRs.shape[0], dataRs.shape[0]))

    ##must return the window to keep it open
    return win
    ## Start Qt event loop unless running in interactive mode.
    if __name__ == '__main__':
        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()