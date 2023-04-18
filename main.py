import numpy as np
from particle_filter.Particle import ParticleFilter, particle_dict, resample_dict
import cv2 as cv
from tqdm import trange
import argparse

def get_opts():
    parser = argparse.ArgumentParser()
    
    # File args
    parser.add_argument('--filepath', type=str, required=True,
                        help='filepath of the video')
    
    # Model args
    parser.add_argument('--N', type=int, default=500, required=True,
                        help='number of particles')
    
    parser.add_argument('--descriptor', type=str, default='surf',
                        choices=['surf', 'sift', 'hog'],# TODO add all descriptor names
                        help='which descriptor to use')
    parser.add_argument('--similarity', type=str, default='bhattacharyya',
                        choices=['bhattacharyya'],
                        help='which similartiy measure to use')
    
    parser.add_argument('--resampling', type=str, default='systemic-resample',
                        choices=['systemic-resample'],
                        help='which resample method to use')
    parser.add_argument('--slicing', type=bool, default=True,
                        help='True : slice images before using the descriptor, False : slice descriptor afterwards)


    return parser.parse_args()

# 
def draw_output_frame(frame : np.array, 
                    descriptor : np.array, 
                    estimate_particle : np.array,
                    color=(255,0,0)):

    output_image = cv.rectangle(frame, 
                        estimate_particle[0] - estimate_particle[6]/2,
                        estimate_particle[3] + estimate_particle[7]/2,
                        color,
                        2)
    # Maybe draw descriptors as well
    cv.imshow('Track Squid', output_image)


def main():
    # get params
    args = get_opts()
    # init params
    filepath = args.filepath
    N = abs(args.N)
    descriptor_fn = getattr(descriptor, 'get_descriptor_'+args.descriptor)
    similarity_fn = getattr(similarity, 'get_similarity_'+arg.similarity)
    # Load video
    n_frames = 100
    cap = cv2.VideoCapture(filepath)
    current_frame = np.array(read(cap))
    last_frame
    # Run YOLOV7 to get bounding boxes
    init_pos, bounding_box = yolo.detect(current_frame)
    # Init Particle filter
    particle_filter =  (N, ConstAccelParticle2DBbox, 1, init_pos, None, None, descriptor_fn, similarity_fn)
    # Get Descriptor 
    descriptor = descriptor.descriptor_fn(current_frame, particle_filter.particles)
    

    # Loop
    for i in trange(1,n_frames): # tqdm bar
        draw_output_frame(current_frame, descriptor, particle_filter.mu)
        cv2.waitKey(0) # Frame by frame
        last_frame = current_frame # Switch frames
        ret, current_frame = cap.read() # Read next frame
        if not ret:
            print("Error : Couldn't read frame")
            quit()
        particle_filter.forward(current_frame) # New pass
        descriptor = particle_filter.descriptor
        tqdm.write(f'Sigma = {particle_filter.sigma}') # Print stuff here
    # End Loop

if __name__ == "__main__":
    main()
