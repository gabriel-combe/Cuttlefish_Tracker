import cv2 
import matplotlib.pyplot as plt
import numpy as np
import time

class HOG():

    def __init__(self, winSize, blockSize = (16, 16), blockStride = (8, 8),  cellSize = (8, 8), nbins = 9):
        assert((winsize[0]-blockSize[0])%blockStride[0]==0 and (winsize[1]-blockSize[1])%blockStride[1]==0)
        assert(blockSize[0]%cellSize[0]==0 and blockSize[1]%cellSize[1]==0)
        self.winSize = (winSize[0] // cellSize[0] * cellSize[0], winSize[1] // cellSize[1] * cellSize[1])
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.hog = cv2.HOGDescriptor(winSize, blockSize, cellSize, cellSize, nbins)

    def expectedF(self):
        return (( (self.winSize[0] - self.blockSize[0]) // self.blockStride[0] ) + 1) * (( (self.winSize[1] - self.blockSize[1]) // self.blockStride[1] ) + 1) * (self.blockSize[0] // self.cellSize[0]) * (self.blockSize[1] // self.cellSize[1]) * self.nbins


    def computeHOGS(self, images : np.ndarray):

        nbImage = images.shape[0]
        expectedSize = self.expectedF()

        histogram = np.empty(shape = (nbImage, expectedSize))
    
        for image in range(nbImage):
            gray = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)
            if self.winSize != gray.shape:
                gray = cv2.resize(gray, self.winSize)
            histogram[image] = self.hog.compute(gray)

        return np.array(histogram)

    def updateHOG(self, winSize, blockSize = (16, 16), blockStride = (8, 8),  cellSize = (8, 8), nbins = 9):
        self.__init__(winSize, blockSize, blockStride,  cellSize, nbins)