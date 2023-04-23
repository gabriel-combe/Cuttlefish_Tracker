from .Slicer import descriptorHOG_slicing, descriptorKP_slicing, image_resize_slicing, image_crop_slicing
from .Similarity import Bhattacharyya_distance
from .Descriptors import HOG, SIFT, ORB, BRISK

slicer_dict = {
    'hog'   : descriptorHOG_slicing,
    'kp'    : descriptorKP_slicing,
    'resize' : image_resize_slicing,
    'crop' : image_crop_slicing,
}

similarity_dict = {
    'bd' : Bhattacharyya_distance
}

descriptor_dict = {
    'hog' : HOG,
    'sift' : SIFT,
    'orb' : ORB,
    'brisk' : BRISK
}