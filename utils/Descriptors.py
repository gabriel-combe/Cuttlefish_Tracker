import cv2
import numpy as np
from skimage import feature

# Entree : Un ndarray correspondant a une image au format BGR
# Sortie : Un tuple 2-Dimension contenant un tuple de keypoint et un numpy array (n, 128) (n=nombre de keypoints) contenant les descripteurs

class HOG():

    def __init__(self, winSize, blockSize = (18, 18), blockStride = (6, 6),  cellSize = (6, 6), nbins = 9, freezeSize = False):
        self.winSize = (winSize[0] // cellSize[0] * cellSize[0], winSize[1] // cellSize[1] * cellSize[1])
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.freezeSize = freezeSize
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
            img = cv2.cvtColor(images[image], cv2.COLOR_BGR2GRAY)
            if self.winSize != img.shape:
                img = cv2.resize(img, self.winSize, interpolation=cv2.INTER_AREA)
            histogram[image] = self.hog.compute(img)

        return np.array(histogram)

    def update(self, winSize):
        if not self.freezeSize:
            self.__init__(winSize, self.blockSize, self.blockStride,  self.cellSize, self.nbins)

class HOGCOLOR():

    def __init__(self, winSize, blockSize = (16, 16), blockStride = (8, 8),  cellSize = (8, 8), nbins = 9, freezeSize = False):
        self.winSize = (winSize[0] // cellSize[0] * cellSize[0], winSize[1] // cellSize[1] * cellSize[1])
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins
        self.freezeSize = freezeSize
        assert((self.winSize[0]-self.blockSize[0])%self.blockStride[0]==0 and (self.winSize[1]-self.blockSize[1])%self.blockStride[1]==0)
        assert(self.blockSize[0]%self.cellSize[0]==0 and self.blockSize[1]%self.cellSize[1]==0)
        self.hog = cv2.HOGDescriptor(self.winSize, self.blockSize, self.cellSize, self.cellSize, self.nbins)

    
    def expectedF(self):
        return (( (self.winSize[0] - self.blockSize[0]) // self.blockStride[0] ) + 1) * (( (self.winSize[1] - self.blockSize[1]) // self.blockStride[1] ) + 1) * (self.blockSize[0] // self.cellSize[0]) * (self.blockSize[1] // self.cellSize[1]) * self.nbins


    def compute(self, images : np.ndarray):

        nbImage = images.shape[0]
        expectedSize = 3*self.expectedF()

        histogram = np.empty(shape = (nbImage, expectedSize))
    
        for image in range(nbImage):
            img = images[image]
            if self.winSize != img.shape:
                img = cv2.resize(img, self.winSize, interpolation=cv2.INTER_AREA)
            histogram[image] = np.concatenate((self.hog.compute(img[:, :, 0]), self.hog.compute(img[:, :, 1]), self.hog.compute(img[:, :, 2])))

        return np.array(histogram)

    def update(self, winSize):
        if not self.freezeSize:
            self.__init__(winSize, self.blockSize, self.blockStride,  self.cellSize, self.nbins)

class LBP():
    def __init__(self, numPoints, radius):
        self.numPoints = numPoints
        self.radius = radius
    
    def compute(self, images : np.ndarray, eps=1e-7):
        hist = []
        for image in images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")
            hist.append(np.histogram(lbp.ravel(), bins=np.arange(0, self.numPoints + 3), range=(0, self.numPoints + 2))[0])

            hist[-1] = hist[-1].astype("float")
            hist[-1] /= (hist[-1].sum() + eps)
        
        return np.array(hist)
    
    def update(self, winSize):
        pass

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