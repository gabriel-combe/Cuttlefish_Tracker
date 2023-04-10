import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def get_descriptor_sift(path):

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return -1


    sift = cv.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)

    return keypoints, descriptors


def get_descriptor_orb(path):

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return -1

    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors


# en test attention
def get_descriptor_hog(path):

    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if image is None:
        return -1
    
    hog = cv.HOGDescriptor()
    keypoints, descriptors = hog.compute(image)
