from .Slicer import Resize, Crop
from .Similarity import Bhattacharyya_distance_sqrt, Bhattacharyya_distance_log, keypoint_matcher
from .Descriptors import HOG, HOGCOLOR, SIFT, ORB, BRISK, LBP

slicer_dict = {
    'resize' : Resize,
    'crop' : Crop,
}

similarity_dict = {
    'bds' : Bhattacharyya_distance_sqrt,
    'bdl' : Bhattacharyya_distance_log,
    'kpm' : keypoint_matcher
}

descriptor_dict = {
    'hog' : HOG,
    'hogcolor' : HOGCOLOR,
    'lbp' : LBP,
    'sift' : SIFT,
    'orb' : ORB,
    'brisk' : BRISK
}