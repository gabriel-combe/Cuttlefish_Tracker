import cv2
import numpy as np

# Bhattacharyya distance between a list of descriptors and a template descriptor
def Bhattacharyya_distance_sqrt(descriptors: np.ndarray, template: np.ndarray):
    # Mean of the vector descriptor
    descriptors_mean = np.mean(descriptors, axis=1)
    template_mean = np.mean(template)

    # Compute the Bhattacharyya coefficient of the descriptor with a template
    bc = np.sum(np.sqrt(descriptors * template), axis=1)

    # Compute the Bhattacharyya distance
    # dist = np.sqrt(1. - (bc * (1./np.sqrt(descriptors_mean * template_mean * template.shape[0]**2))))

    # Use this distance if there's issue with floating point number
    # Or if a descriptor full of zeros may occur
    dist = np.sqrt(np.maximum(0., 1. - (bc * (1./(np.sqrt(descriptors_mean * template_mean * template.shape[1]**2) + np.finfo(float).tiny)))))

    return dist

# Bhattacharyya distance between a list of descriptors and a template descriptor
def Bhattacharyya_distance_log(descriptors: np.ndarray, template: np.ndarray):
    # Mean of the vector descriptor
    descriptors_mean = np.mean(descriptors, axis=1)
    template_mean = np.mean(template)

    # Compute the Bhattacharyya coefficient of the descriptor with a template
    bc = np.sum(np.sqrt(descriptors * template), axis=1)

    # Compute the Bhattacharyya distance
    # dist = np.sqrt(1. - (bc * (1./np.sqrt(descriptors_mean * template_mean * template.shape[0]**2))))

    # Use this distance if there's issue with floating point number
    # Or if a descriptor full of zeros may occur
    dist = -np.log(np.minimum(1., bc * (1./(np.sqrt(descriptors_mean * template_mean * template.shape[1]**2) + np.finfo(float).tiny))) + np.finfo(float).tiny)

    return dist

# Keypoint matcher similarity
def keypoint_matcher(descriptors: np.ndarray, template: np.ndarray):
    coeff = []
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i, desc in enumerate(descriptors):
        matches = bf.match(desc[1], template[0][1])
        if desc and matches:
            coeff.append(len(matches)/len(desc[1]))
        else:
            coeff.append(0)

    return np.array(coeff)