import numpy as np

def Bhattacharyya_distance(descriptors: np.ndarray, template: np.ndarray):
    # Mean of the vector descriptor
    descriptors_mean = np.mean(descriptors, axis=1)
    template_mean = np.mean(template)

    # Compute the Bhattacharyya coefficient of the descriptor with a template
    bc = np.sum(np.sqrt(descriptors * template), axis=1)

    # Compute the Bhattacharyya distance
    dist = np.sqrt(1. - (bc * (1./np.sqrt(descriptors_mean * template_mean * template.shape[0]**2))))

    # Use this distance if there's issue with floating point number
    # Or if a descriptor full of zeros may occur
    #dist = np.sqrt(np.maximum(0., 1. - (bc * (1./(np.sqrt(descriptors_mean * template_mean * template.shape[0]**2) + np.finfo(float).tiny)))))

    return dist