import cv2
import numpy as np

# Similarity class template
class Similarity():
    def __init__(self):
        pass

    def computeSimilarity(self, descriptors: np.ndarray, template: np.ndarray) -> np.ndarray:
        pass

# Bhattacharyya distance between a list of descriptors and a template descriptor
class Bhattacharyya_sqrt(Similarity):
    def computeSimilarity(self, descriptors: np.ndarray, template: np.ndarray) -> np.ndarray:
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
class Bhattacharyya_log(Similarity):
    def computeSimilarity(self, descriptors: np.ndarray, template: np.ndarray) -> np.ndarray:
        # Mean of the vector descriptor
        descriptors_mean = np.mean(descriptors, axis=1)
        template_mean = np.mean(template)

        # Compute the Bhattacharyya coefficient of the descriptor with a template
        bc = np.sum(np.sqrt(descriptors * template), axis=1)

        # Compute the Bhattacharyya distance
        # dist = -np.log(bc * (1./np.sqrt(descriptors_mean * template_mean * template.shape[1]**2)))

        # Use this distance if there's issue with floating point number
        # Or if a descriptor full of zeros may occur
        dist = -np.log(np.minimum(1., bc * (1./(np.sqrt(descriptors_mean * template_mean * template.shape[1]**2) + np.finfo(float).tiny))) + np.finfo(float).tiny)

        return dist

# Cosine similarity function between a set of descriptor and a template
class Cosine_Similarity(Similarity):
    def computeSimilarity(self, descriptors: np.ndarray, template: np.ndarray) -> np.ndarray:
        descriptors_normalized = descriptors / np.linalg.norm(descriptors, axis=1)[:, np.newaxis]
        template_normalized = template / np.linalg.norm(template, axis=1)[:, np.newaxis]

        similarity = np.sum(descriptors_normalized*template_normalized, axis=1)

        return 1 - similarity

# Kullback Leibler Divergence function between a set of descriptor and a template
class Kullback_Leibler_Divergence(Similarity):
    def computeSimilarity(self, descriptors: np.ndarray, template: np.ndarray) -> np.ndarray:
        indices_nonzero_template = template != 0

        similarity = []

        for desc in descriptors:
            indices_nonzero_descriptors = desc != 0 & indices_nonzero_template[0]
            similarity.append(np.sum(desc[indices_nonzero_descriptors] * np.log2(desc[indices_nonzero_descriptors] / template[indices_nonzero_template])))

        return np.array(similarity)

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