import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Entree : Un ndarray correspondant a une image au format BGR
# Sortie : Un tuple 2-Dimension contenant un tuple de keypoint et un numpy array (n, 128) (n=nombre de keypoints) contenant les descripteurs

def get_descriptor_sift(images):

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)

    return keypoints, descriptors


def get_descriptor_orb(image):

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image_gray, None)

    return keypoints, descriptors


def get_descriptor_brisk(image):

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    brisk = cv.BRISK_create()
    keypoints, descriptors = brisk.detectAndCompute(image_gray, None)

    return keypoints, descriptors

def get_descriptors_sift(image, particle_array, ix=0, iy=3, iw=6, ih=7):

    # Initialisation de sift 
    sift = cv.SIFT_create()

    # Conversion en gris, sift ne peut travaille que sur image en niveau de gris
    cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # création d'un array numpy de la même taille quie particle_array initialisé à 0.
    keypoints_descriptors = [0]*particle_array.shape[0]

    # calcule des keypoints et descripteurs pour chaque box correspondant à une particule
    for particle in range(particle_array.shape[0]):

        x, y, w, h = particle_array[particle, ix], particle_array[particle, iy], particle_array[particle, iw], particle_array[particle, ih]
        keypoints_descriptors[particle] = sift.detectAndCompute(image[x-int(w/2):x+int(w/2), y-int(h/2):y+int(h/2)], None)

    return keypoints_descriptors


# Entree : numpy array : liste d'image au format BGR, Sortie : numpy array : histogram
# def get_descriptor_hog(images, blockSize = (16, 16), blockStride = (8, 8),  cellSize = (8, 8), nbins = 9):
def get_descriptor_hog(images):

    histograms = []

    # Calcule la taille inférieur tel qu'elle soit multiple de 8, 8 = cellsize
    wsize = (int(images[0].shape[0] / cellSize[0]) * cellSize[0], int(images[0].shape[1] / cellSize[1]) * cellSize[1])

    
    hog = cv.HOGDescriptor() # Initialisation par défaut de HOG
    
    # Initialisation avec wsize custom de hog, nécessite des tests et l'autre header pour la fonction
    # hog = cv2.HOGDescriptor(wsize, blockSize, blockStride, cellSize, nbins)

    # Parcours des images de la list numpy image
    for image in images:
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # Conversion du format BGR à gris nécessaire à HOG

        if (image.shape[:2]!=wsize):            # resize des images sur un multiple inférieur de 8
            gray = cv2.resize(gray, wsize)

        histogram += [hog.compute(gray)] # Calcule des histograms

    return histogram




