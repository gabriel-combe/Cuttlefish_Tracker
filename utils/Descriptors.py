import cv2
import numpy as np

# Entree : Un ndarray correspondant a une image au format BGR
# Sortie : Un tuple 2-Dimension contenant un tuple de keypoint et un numpy array (n, 128) (n=nombre de keypoints) contenant les descripteurs

class HOG():

    def __init__(self, winSize, blockSize = (16, 16), blockStride = (8, 8),  cellSize = (8, 8), nbins = 9):
        self.winSize = (winSize[0] // cellSize[0] * cellSize[0], winSize[1] // cellSize[1] * cellSize[1])
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        assert((self.winSize[0]-self.blockSize[0])%self.blockStride[0]==0 and (self.winSize[1]-self.blockSize[1])%self.blockStride[1]==0)
        assert(self.blockSize[0]%self.cellSize[0]==0 and self.blockSize[1]%self.cellSize[1]==0)
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.cellSize, self.cellSize, self.nbins)

    
    def expectedF(self):
        return (( (self.winSize[0] - self.blockSize[0]) // self.blockStride[0] ) + 1) * (( (self.winSize[1] - self.blockSize[1]) // self.blockStride[1] ) + 1) * (self.blockSize[0] // self.cellSize[0]) * (self.blockSize[1] // self.cellSize[1]) * self.nbins


    def compute(self, images : np.ndarray):

        nbImage = images.shape[0]
        expectedSize = self.expectedF()

        histogram = np.empty(shape = (nbImage, expectedSize))
    
        for image in range(nbImage):
            gray = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)
            if self.winSize != gray.shape:
                gray = cv2.resize(gray, self.winSize)
            histogram[image] = self.hog.compute(gray)

        return np.array(histogram)

    def update(self, winSize):
        self.__init__(winSize, self.blockSize, self.blockStride,  self.cellSize, self.nbins)

class SIFT():

    def __init__ (self, nfeatures: int):
        self.sift = cv2.SIFT_create(nfeatures=nfeatures)

    def compute(self, images):

        keypoints_descriptors = []

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints_descriptors += [self.sift.detectAndCompute(gray, None)]

        return keypoints_descriptors

import matplotlib.pyplot as plt
class ORB():

    def __init__ (self, nfeatures: int):
        self.orb = cv2.ORB_create(nfeatures=nfeatures)

    def compute(self, images):

        keypoints_descriptors = []

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints_descriptors += [self.orb.detectAndCompute(gray, None)]

        return keypoints_descriptors
    
    def update(self, dummy):
        pass


class BRISK():

    def __init__ (self, nfeatures: int):
        self.brisk = cv2.BRISK_create(nfeatures=nfeatures)

    def compute(self, images):

        keypoints_descriptors = []

        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints_descriptors += [self.brisk.detectAndCompute(gray, None)]

        return keypoints_descriptors