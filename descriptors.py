import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def get_descriptor_sift(image):

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
