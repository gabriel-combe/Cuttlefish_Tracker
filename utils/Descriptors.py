import cv2
import matplotlib.pyplot as plt
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

    def __init__(self):
        self

def get_descriptor_sift(images):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)

    return keypoints, descriptors


def get_descriptor_orb(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)

    return keypoints, descriptors


def get_descriptor_brisk(image):

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brisk = cv2.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(image_gray, None)

    return keypoints, descriptors

def get_descriptors_sift(image, particle_array, ix=0, iy=3, iw=6, ih=7):

    # Initialisation de sift 
    sift = cv2.SIFT_create()

    # Conversion en gris, sift ne peut travaille que sur image en niveau de gris
    cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # création d'un array numpy de la même taille quie particle_array initialisé à 0.
    keypoints_descriptors = [0]*particle_array.shape[0]

    # calcule des keypoints et descripteurs pour chaque box correspondant à une particule
    for particle in range(particle_array.shape[0]):

        x, y, w, h = particle_array[particle, ix], particle_array[particle, iy], particle_array[particle, iw], particle_array[particle, ih]
        keypoints_descriptors[particle] = sift.detectAndCompute(image[x-int(w/2):x+int(w/2), y-int(h/2):y+int(h/2)], None)

    return keypoints_descriptors