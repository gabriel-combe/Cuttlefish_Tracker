import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Entree : Un ndarray correspondant a une image au format BGR
# Sortie : Un tuple 2-Dimension contenant un ndarray contenant les keypoints et un ndarray contenant les descripteurs

def get_descriptor_sift(image, particleArray=None):
    # ParticleArray = ndarray contenant les particles associes a l'image. / Partie pas encore finie

    # Abscisse, Ordonnee, Largeur, Hauteur
    indices = 0
    for particle in particleArray:
        posx, posy, width, height = particle[0], particle[3], particle[-2], particle[-1]
        originx, originy = posx-(width/2), posy-(height/2)
        indices =

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def get_descriptor_orb(image):

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors




# en test attention
def get_descriptor_hog(image):
    
    hog = cv.HOGDescriptor()
    keypoints, descriptors = hog.compute(image)
