import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Entree : Un ndarray correspondant a une image au format BGR
# Sortie : Un tuple 2-Dimension contenant un ndarray contenant les keypoints et un ndarray contenant les descripteurs

def get_descriptor_sift(image, particleArray=None):
    # particleArray = ndarray contenant les particles associes a l'image. / Partie pas encore finie

    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    keypoints, descriptors = np.empty(), np.empty()

    for particle in particleArray:
        x, y, horizontal, vertical = particle[0], particle[3], particle[6], particle[7]
        keypoi

    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors