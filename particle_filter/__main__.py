import numpy as np
from ParticleFilter import *
import cv2 as cv


def initialize():
    print("Initialisation")

def main():
    # init params
    N = 500
    descriptor_fn = getattr(descriptor, 'get_descriptor_sift')
    similarity_fn = getattr(similarity, 'get_similarity_bhattacharyya')
    # Load video
    n_frames = 100
    cap = cv2.VideoCapture(file)
    current_frame = np.array(read(cap))
    last_frame
    # Run YOLOV7 to get bounding boxes and descriptor
    init_pos = yolo.detect(current_frame)

    # Init Particle filter
    particle_filter =  (N, ConstAccelParticle2DBbox, 1, init_pos, None, None, descriptor_fn, similarity_fn)
    # Get Descriptor    
    descriptor = descriptor.descriptor_fn(current_frame, particle_filter.particles)


    
    # Loop
    for i in range (1,n_frames):
        last_frame = current_frame
        if cap.isOpened():
            current_frame = np.array(read(cap))
        else :
            print("Error reading file")
            quit()
    
        


if __name__ == "__main__":
    main()



