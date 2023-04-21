from .Slicer import descriptorHOG_slicing, descriptorKP_slicing, image_resize_slicing, image_crop_slicing

slicer_dict = {
    'HOG'   : descriptorHOG_slicing,
    'KP'    : descriptorKP_slicing,
    'Resize' : image_resize_slicing,
    'Crop' : image_crop_slicing,
}