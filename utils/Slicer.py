import numpy as np
from typing import Tuple, List

# Get the slice of the HOG descriptor of an image based on particles position and Bbox
# Compute the slice with the same Bbox size but with the template_particle position
# and the descriptor of the template_image
def descriptorHOG_slicing(
    particles: np.ndarray, descriptor: np.ndarray, 
    template_particle: np.ndarray, template_descriptor: np.ndarray,
    winSize: Tuple[int, int], blockSize: Tuple[int, int] =(16, 16), blockStride: Tuple[int, int] =(8, 8),
    cellSize: Tuple[int, int] =(8, 8), nbins: int =9) -> List[np.ndarray]:

    sliced_descriptor = []

    # Compute padding width and height
    pad_width = (1 + np.max(particles[:, :, 6])//(2 * blockSize[0])) * (blockSize[0]//cellSize[0]) * nbins
    pad_height = (1 + np.max(particles[:, :, 7])//(2 * blockSize[1])) * (blockSize[1]//cellSize[1]) * nbins

    print(pad_width, "  ", pad_height)

    descriptor_width = winSize[0]//cellSize[0]
    descriptor_height = winSize[1]//cellSize[1]

    width_block = ((winSize[0] - blockSize[0]) // blockStride[0]) + 1
    height_block = ((winSize[1] - blockSize[1]) // blockStride[1]) + 1

    subDescriptorSize = (blockSize[0]//cellSize[0]) * (blockSize[1]//cellSize[1]) * nbins

    for p in particles:
        x_start = int(p[0, 0])
        y_start = int(p[0, 3])
        x_end = int(p[0, 0] + p[0, 6])
        y_end = int(p[0, 3] + p[0, 7])

        block_x_start = 0 if x_start - blockSize[0] < 0 else ((x_start - blockSize[0])//blockStride[0]) + 1
        block_y_start = 0 if y_start - blockSize[1] < 0 else ((y_start - blockSize[1])//blockStride[1]) + 1

        cell_start = (x_start + y_start * winSize[0])
        cell_end = (x_end + y_end * winSize[0])
        step_size = (winSize[0] - (x_end - x_start))

        #print(cell_start, "    ", cell_end, "    ", step_size)

        x_template_start = int(template_particle[0])
        y_template_start = int(template_particle[3])
        x_template_end = int(template_particle[0] + p[0, 6])
        y_template_end = int(template_particle[3] + p[0, 7])

        cell_start_template = (x_template_start + y_template_start * winSize[0])
        cell_end_template = (x_template_end + y_template_end * winSize[0])
        step_size_template = (winSize[0] - (x_template_end - x_template_start))

        sliced_descriptor.append(np.array([
            descriptor[cell_start*nbins:(cell_end+1)*nbins:step_size*nbins],
            template_descriptor[cell_start_template*nbins:(cell_end_template+1)*nbins:step_size_template*nbins]
        ]))

        #print(sliced_descriptor[-1][0].shape)
        #print(sliced_descriptor[-1][1].shape)
    
    return sliced_descriptor

# Get the slice of the descriptor of an image based on particles position and Bbox
# Compute the slice with the same Bbox size but with the template_particle position
# and the descriptor of the template_image
def descriptorKP_slicing(
    particles: np.ndarray, descriptor: np.ndarray, 
    template_particle: np.ndarray, template_descriptor: np.ndarray) -> np.ndarray:

    pass

# Get the slice of an image based on particles position and Bbox
# Rescale each image patch to the size of the Bbox of the template_particle
# Compute the slice of the template_image with the template_particle position and Bbox
def image_resize_slicing(
    particles: np.ndarray, image: np.ndarray,
    template_particle: np.ndarray) -> List[np.ndarray]:
    
    sliced_image = []

    # Compute padding width and height
    pad_width = np.max(particles[:, :, 6])//2
    pad_height = np.max(particles[:, :, 7])//2
    template_pad_width = template_particle[7]//2
    template_pad_height = template_particle[6]//2

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
        final_patch = cv2.resize(patch, (template_particle[6], template_particle[7]))

        sliced_image.append(final_patch)

    return np.array(sliced_image)

def image_crop_slicing(
    particles: np.ndarray, image: np.ndarray,
    template_particle: np.ndarray) -> List[np.ndarray]:
    
    sliced_image = []

    # Compute padding width and height
    pad_width = np.max(particles[:, :, 6])//2
    pad_height = np.max(particles[:, :, 7])//2
    template_pad_width = template_particle[7]//2
    template_pad_height = template_particle[6]//2

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




import cv2
import time

image = cv2.imread("./utils/Squid_colors_2.jpg", cv2.IMREAD_COLOR)
cv2.imshow("original", image)
print(image.shape)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

winSize = (gray.shape[1], gray.shape[0])
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
descriptor = hog.compute(gray)

print(gray.shape)

p0 = np.array([350, 0, 0, 350, 0, 0, 160, 180])
p1 = np.array([
    [[360, 0, 0, 345, 0, 0, 163, 170]],
    [[362, 0, 0, 342, 0, 0, 165, 172]],
    [[380, 0, 0, 300, 0, 0, 160, 176]],
    [[358, 0, 0, 348, 0, 0, 158, 169]],
    [[345, 0, 0, 350, 0, 0, 167, 170]]
])

print(descriptor.shape)

start = time.perf_counter()
patches = image_resize_slicing(p1, image, p0)
print(time.perf_counter() - start)

start = time.perf_counter()
sliced_descriptor = descriptorHOG_slicing(p1, descriptor, p0, descriptor, winSize)
print(time.perf_counter() - start)

for i, patch in enumerate(patches):
    cv2.imshow(f"{i}", patch)

cv2.waitKey(0)
cv2.destroyAllWindows()