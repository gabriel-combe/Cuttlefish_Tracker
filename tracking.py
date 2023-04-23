import cv2
import argparse
import numpy as np
from tqdm import trange
from particle_filter import ParticleFilter, particle_dict, resample_dict
from utils import slicer_dict, similarity_dict, descriptor_dict
from detect_init import Model

def get_opts():
    parser = argparse.ArgumentParser()
    
    # File args
    parser.add_argument('--filepath', type=str, required=True,
                        help='filepath of the video')
    
    # Particle Filter args
    parser.add_argument('--N', type=int, default=500,
                        help='number of particles')
    parser.add_argument('--particle', type=str, defautl='cap2Dbb',
                        choices=['cap2Dfbb', 'cap2Dbb'],
                        help='which particle structure to use')
    parser.add_argument('--descriptor', type=str, default='hog',
                        choices=['brisk', 'sift', 'hog', 'orb', 'color'],
                        help='which descriptor to use')
    parser.add_argument('--similarity', type=str, default='bhattacharyya',
                        choices=['bhattacharyya'],
                        help='which similartiy measure to use')
    parser.add_argument('--resampling', type=str, default='systematic',
                        choices=['systematic', 'residual', 'stratified', 'multinomial'],
                        help='which resample method to use')
    parser.add_argument('--slicer', type=str, default='resize',
                        choices=['hog', 'kp', 'resize', 'crop'],
                        help='which slicer to use')
    
    # Model args
    parser.add_argument('--weights', nargs='+', type=str, default='weights/cuttlefish_best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')


    return parser.parse_args()

# 
def draw_output_frame(frame : np.array, 
                    descriptor : np.array, 
                    estimate_particle : np.array,
                    color=(255,0,0)):

    output_image = cv.rectangle(frame, 
                        estimate_particle[0] - estimate_particle[6]/2,
                        estimate_particle[3] - estimate_particle[7]/2,
                        color,
                        2)
    # Maybe draw descriptors as well
    cv.imshow('Track Cuttlefish', output_image)
    

if __name__ == "__main__":
    # get params
    args = get_opts()

    # init params
    N = abs(args.N)
    model = Model(
        weights=args.weights, device=args.device, img_size=args.img_size,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres)
    descriptor = descriptor_dict[args.descriptor]
    similarity = similarity_dict[args.similarity]
    resampling = resample_dict[args.resampling]
    slicer = slicer_dict[args.slicer]
    particle_struct = particle_dict[args.particle]

    # Load video
    cap = cv2.VideoCapture(args.filepath)

    # Read first frame
    ret, current_frame = cap.read()
    if not ret:
        print("Error : Couldn't read frame")
        quit()

    # Run YOLOV7 to get bounding boxes
    prediction = model.detect(current_frame)

    # Init Particle filter
    particle_filter =  (N, particle_struct, 1, init_pos, None, None, slicer, descriptor, similarity, resampling)
    
    # Get Descriptor 
    descriptor = descriptor.descriptor_fn(current_frame, particle_filter.particles)
    

    # Loop
    for i in trange(1,n_frames):
        draw_output_frame(current_frame, descriptor, particle_filter.mu)


        ret, current_frame = cap.read() # Read next frame

        if not ret:
            print("Error : Couldn't read frame")
            quit()

        particle_filter.forward(current_frame) # New pass

        cv2.waitKey(0)
    # End Loop

    cv2.destroyAllWindows()
