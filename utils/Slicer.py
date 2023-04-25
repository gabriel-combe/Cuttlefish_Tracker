import cv2
import numpy as np
from typing import Tuple, List

# Get the slice of an image based on particles position and Bbox
# Rescale each image patch to the size of the Bbox of the template_particle
def image_resize_slicing(
    particles: np.ndarray, image: np.ndarray,
    template_particle: np.ndarray) -> np.ndarray:

    sliced_image = []

    # Compute padding width and height
    pad_width = int(np.max(particles[:, :, 6])//2)
    pad_height = int(np.max(particles[:, :, 7])//2)

    # Pad images to take care of Bbox that get outside of the image
    image_pad = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)

    for p in particles:
        # Compute Start and End point to determine the patch of the image to slice
        x_start = int(p[0, 0])
        y_start = int(p[0, 3])
        x_end = int(p[0, 0] + p[0, 6])
        y_end = int(p[0, 3] + p[0, 7])

        # Slice the image based on particles position
        patch = image_pad[y_start:y_end, x_start:x_end]

        # Resize the patch to match the size of the template patch
        final_patch = cv2.resize(patch, (int(template_particle[6]), int(template_particle[7])))

        sliced_image.append(final_patch)

    return np.array(sliced_image)

# Get the slice of an image based on particles position and Bbox
# Crop each image patch to the size of the Bbox of the template_particle
def image_crop_slicing(
    particles: np.ndarray, image: np.ndarray,
    template_particle: np.ndarray) -> np.ndarray:
    
    sliced_image = []

    # Compute padding width and height
    template_pad_width = int(template_particle[7]//2)
    template_pad_height = int(template_particle[6]//2)

    # Pad images to take care of Bbox that get outside of the image
    image_pad = np.pad(image, ((template_pad_height, template_pad_height), (template_pad_width, template_pad_width), (0, 0)), mode='constant', constant_values=0)

    for p in particles:
        # Compute Start and End point to determine the patch of the image to slice
        x_start = int(p[0, 0])
        y_start = int(p[0, 3])
        x_end = int(p[0, 0] + template_particle[6])
        y_end = int(p[0, 3] + template_particle[7])

        # Crop the patch to the size of the template patch
        final_patch = image_pad[y_start:y_end, x_start:x_end]

        sliced_image.append(final_patch)

    return np.array(sliced_image)

# Get the slice of an image based on the template particle position and Bbox
def template_image_slicing(
    template_image: np.ndarray,
    template_particle: np.ndarray) -> np.ndarray:
    
    # Compute padding width and height
    template_pad_width = int(template_particle[7]//2)
    template_pad_height = int(template_particle[6]//2)

    # Pad images to take care of Bbox that get outside of the image
    template_image_pad = np.pad(template_image, ((template_pad_height, template_pad_height), (template_pad_width, template_pad_width), (0, 0)), mode='constant', constant_values=0)

    # Compute Start and End point to determine the patch of the image to slice
    x_start = int(template_particle[0])
    y_start = int(template_particle[3])
    x_end = int(template_particle[0] + template_particle[6])
    y_end = int(template_particle[3] + template_particle[7])

    # Slice the image based on particles position
    patch = template_image_pad[y_start:y_end, x_start:x_end]

    return patch


#####################################################
######## Attempt to slice the descriptor ############
#####################################################

# Get the slice of the HOG descriptor of an image based on particles position and Bbox
# Compute the slice with the same Bbox size but with the template_particle position
# and the descriptor of the template_image
def descriptorHOG_slicing(
    particles: np.ndarray, descriptor: np.ndarray, 
    template_particle: np.ndarray, template_descriptor: np.ndarray,
    descriptor_obj) -> List[np.ndarray]:

    winSize = descriptor_obj.winSize
    blockSize = descriptor_obj.blockSize
    blockStride = descriptor_obj.blockStride
    cellSize = descriptor_obj.cellSize
    nbins = descriptor_obj.nbins

    sliced_descriptor = []

    # Compute padding width and height
    pad_width = int(1 + np.max(particles[:, :, 6])//(2 * blockSize[0]))
    pad_height = int(1 + np.max(particles[:, :, 7])//(2 * blockSize[1]))

    width_block = ((winSize[0] - blockSize[0]) // blockStride[0]) + 1
    height_block = ((winSize[1] - blockSize[1]) // blockStride[1]) + 1

    nbCellPerBlock = (blockSize[0]//cellSize[0]) * (blockSize[1]//cellSize[1])

    descriptor = np.reshape(descriptor, (height_block, width_block, nbCellPerBlock, nbins))
    template_descriptor = np.reshape(template_descriptor, (height_block, width_block, nbCellPerBlock, nbins))

    descriptor_pad = np.pad(descriptor, ((pad_height, pad_height), (pad_width, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=0)
    template_descriptor_pad = np.pad(template_descriptor, ((pad_height, pad_height), (pad_width, pad_width), (0, 0), (0, 0)), mode='constant', constant_values=0)

    for p in particles:
        x_start = int(p[0, 0])
        y_start = int(p[0, 3])
        x_end = int(p[0, 0] + p[0, 6])
        y_end = int(p[0, 3] + p[0, 7])

        block_x_start = 0 if x_start - blockSize[0] < 0 else ((x_start - blockSize[0])//blockStride[0]) + 1
        block_y_start = 0 if y_start - blockSize[1] < 0 else ((y_start - blockSize[1])//blockStride[1]) + 1

        block_x_end = ((x_end + blockSize[0])//blockStride[0]) - 2
        block_y_end = ((y_end + blockSize[1])//blockStride[1]) - 2

        x_template_start = int(template_particle[0])
        y_template_start = int(template_particle[3])
        x_template_end = int(template_particle[0] + p[0, 6])
        y_template_end = int(template_particle[3] + p[0, 7])

        block_x_start_template = 0 if x_template_start - blockSize[0] < 0 else ((x_template_start - blockSize[0])//blockStride[0]) + 1
        block_y_start_template = 0 if y_template_start - blockSize[1] < 0 else ((y_template_start - blockSize[1])//blockStride[1]) + 1

        block_x_end_template = ((x_template_end + blockSize[0])//blockStride[0]) - 2
        block_y_end_template = ((y_template_end + blockSize[1])//blockStride[1]) - 2

        if block_x_end - block_x_start < block_x_end_template - block_x_start_template:
            if abs(block_x_start - block_x_start_template) > abs(block_x_end - block_x_end_template):
                block_x_start -= 1
            else:
                block_x_end += 1
        elif block_x_end - block_x_start > block_x_end_template - block_x_start_template:
            if abs(block_x_start - block_x_start_template) > abs(block_x_end - block_x_end_template):
                block_x_start_template -= 1
            else:
                block_x_end_template += 1
            
        if block_y_end - block_y_start < block_y_end_template - block_y_start_template:
            if abs(block_y_start - block_y_start_template) > abs(block_y_end - block_y_end_template):
                block_y_start -= 1
            else:
                block_y_end += 1
        elif block_y_end - block_y_start > block_y_end_template - block_y_start_template:
            if abs(block_y_start - block_y_start_template) > abs(block_y_end - block_y_end_template):
                block_y_start_template -= 1
            else:
                block_y_end_template += 1

        descriptor_patch = descriptor_pad[block_y_start:block_y_end+1, block_x_start:block_x_end+1]
        template_descriptor_patch = template_descriptor_pad[block_y_start_template:block_y_end_template+1, block_x_start_template:block_x_end_template+1]

        sliced_descriptor.append(np.array([
            descriptor_patch.flatten(),
            template_descriptor_patch.flatten()
        ]))
    
    return sliced_descriptor