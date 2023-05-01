from .Slicer import Resize, Crop
from .Similarity import Bhattacharyya_distance_sqrt, Bhattacharyya_distance_log, Cosine_Similarity, Kullback_Leibler_Divergence, keypoint_matcher
from .Descriptors import HOG, HOGCOLOR, HOGCASCADE, HOGCASCADELBP, SIFT, ORB, BRISK, LBP

slicer_dict = {
    'resize' : Resize,
    'crop' : Crop,
}

similarity_dict = {
    'bds' : Bhattacharyya_distance_sqrt,
    'bdl' : Bhattacharyya_distance_log,
    'cos' : Cosine_Similarity,
    'dkl' : Kullback_Leibler_Divergence,
    'kpm' : keypoint_matcher
}

descriptor_dict = {
    'hog' : HOG,
    'hogcolor' : HOGCOLOR,
    'hogcascade' : HOGCASCADE,
    'hogcascadelbp' : HOGCASCADELBP,
    'lbp' : LBP,
    'sift' : SIFT,
    'orb' : ORB,
    'brisk' : BRISK
}