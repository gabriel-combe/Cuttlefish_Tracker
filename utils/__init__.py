from .Slicer import descriptorHOG_slicing, descriptorKP_slicing, image_resize_slicing, image_crop_slicing
from .Similarity import Bhattacharyya_distance_image, Bhattacharyya_distance_descriptor
from .Descriptors import HOG, SIFT, ORB, BRISK

slicer_dict = {
    'hog'   : descriptorHOG_slicing,
    'kp'    : descriptorKP_slicing,
    'resize' : image_resize_slicing,
    'crop' : image_crop_slicing,
}

similarity_image_dict = {
    'bd' : Bhattacharyya_distance_image
}

similarity_descriptor_dict = {
    'bd' : Bhattacharyya_distance_descriptor
}

descriptor_dict = {
    'hog' : HOG,
    'sift' : SIFT,
    'orb' : ORB,
    'brisk' : BRISK
}