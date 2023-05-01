import cv2
import argparse
import numpy as np
from particle_filter import ParticleFilter, particle_dict, resample_dict
from utils import descriptor_dict, similarity_dict, slicer_dict
from detect_init import Model
from typing import Tuple

def get_opts():
    parser = argparse.ArgumentParser()
    
    # File args
    parser.add_argument('--filepath', type=str, required=True, help='filepath of the video')
    parser.add_argument('--scale-factor', type=float, default=1.,
                        help='Scale used to resize the frames of the video')
    
    # Particle Filter args
    parser.add_argument('--N', type=int, default=500,
                        help='number of particles')
    parser.add_argument('--particle', type=str, default='cap2Dbb',
                        choices=['cap2Dfbb', 'cap2Dbb', 'ppp2Dbb'],
                        help='which particle structure to use')
    parser.add_argument('--descriptor', type=str, default='hog',
                        choices=['hog', 'hogcolor', 'hogcascade', 'hogcascadelbp', 'lbp'],
                        help='which descriptor to use')
    parser.add_argument('--similarity', type=str, default='bd',
                        choices=['bds', 'bdl', 'cos', 'dkl'],
                        help='which similartiy measure to use')
    parser.add_argument('--resampling', type=str, default='systematic',
                        choices=['systematic', 'residual', 'stratified', 'multinomial'],
                        help='which resample method to use')
    parser.add_argument('--slicer', type=str, default='resize',
                        choices=['resize', 'crop'],
                        help='which slicer to use')
    parser.add_argument('--alpha', type=float, default=1.2, help='Scaling factor for the search area')
    parser.add_argument('--resample-factor', type=float, default=1./4., help='Factor used to compute the resampling threshold')
    
    # Descriptor args
    parser.add_argument('--nb-features', type=int, default=4000, help='Max number of feature for keypoint descriptors')
    parser.add_argument('--desc-size', nargs=2, type=int, default=None, help='Fix size used force all patch to have the same size for each frame')
    parser.add_argument('--lbp-radius', type=int, default=1, help='Radius of the LBP')
    parser.add_argument('--lbp-nbpoints', type=int, default=8, help='Number of point used for the LBP')
    
    # Model args
    parser.add_argument('--weights', nargs='+', type=str, default='weights/cuttlefish_best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')

    # Video args
    parser.add_argument('--save-video', action='store_true', help='Save a video')
    parser.add_argument('--save-path', type=str, default='results', help='Path to save the video')

    # Seed for random generator
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random generator')


    return parser.parse_args()

# Draw Bbox
def draw_output_frame(frame : np.array, estimate_particle : np.array, color=(0,255,0)):
    output_frame = frame
    for p in estimate_particle:
        output_frame = cv2.rectangle(
            output_frame, 
            (int(p[0] - p[6]), int(p[3] - p[7])),
            (int(p[0] + p[6]), int(p[3] + p[7])),
            color,
            thickness=2)
                    
    return output_frame

# Draw each particles
def draw_output_particles(frame : np.array, particles : np.array,color=(0,0,255)):
    output_frame = frame
    for particle in particles[:, 0]:
        output_image = cv2.circle(output_frame, (int(particle[0]), int(particle[3])), radius=1, color=color, thickness=1)
    return output_image

# Draw the mean particle
def draw_output_mean_particule(frame : np.array, mean_particles : np.array, color=(255,0,0)):
    output_frame = frame
    for p in mean_particles:
        output_image = cv2.circle(output_frame, (int(p[0]), int(p[3])), radius=3, color=color, thickness=3)
    return output_image

# Draw the search area
def draw_search_area(frame : np.array, mean_particles : np.array, search_area : np.array, color=(0,0,255)):
    output_frame = frame
    for i, p in enumerate(mean_particles):
        output_frame = cv2.rectangle(
            output_frame, 
            (int(p[0] - search_area[i, 0]), int(p[3] - search_area[i, 1])), 
            (int(p[0] + search_area[i, 0]), int(p[3] + search_area[i, 1])), 
            color, thickness=2)
    return output_frame


# Select a cuttlefish to track based on confidence and distance from the center of the image
def cuttlefish_picker_WSE(init_frame, conf, cuttlefish):
    error = (1-conf) * np.sqrt(((init_frame.shape[1]//2)-cuttlefish[:, 0])**2 + ((init_frame.shape[0]//2)-cuttlefish[:, 1])**2)
    index = np.argmin(error)
    return (conf[index], cuttlefish[index])

# Select a random cuttlefish to track 
def cuttlefish_picker_random(init_frame, conf, cuttlefish):
    index = np.random.default_rng().integers(low=0, high=conf.shape[0])
    return (conf[index], cuttlefish[index])




if __name__ == "__main__":
    # get params
    args = get_opts()

    # init params
    N = abs(args.N)
    seed = args.seed
    stop = False

    # init Model
    model = Model(
        weights=args.weights, device=args.device, img_size=args.img_size,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres)

    # Load video
    cap = cv2.VideoCapture(args.filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Select a cuttlefish to track
    conf = []
    while not len(conf):
        # Read first frame
        ret, current_frame = cap.read()
        if not ret:
            print("Error : Couldn't read frame")
            quit()

        # Resize if video_size is not None
        if args.scale_factor != 1.:
            current_frame = cv2.resize(current_frame, (int(current_frame.shape[1]*args.scale_factor), int(current_frame.shape[0]*args.scale_factor)), interpolation=cv2.INTER_AREA)

        # Save image dimensions
        img_size = (current_frame.shape[1], current_frame.shape[0])

        # Run YOLOV7 to get bounding boxes
        conf, cuttlefish = model.detect(current_frame)

        if(len(conf)):
            tracked_conf, tracked_cuttlefish = cuttlefish_picker_WSE(current_frame, conf, cuttlefish)
            # tracked_conf, tracked_cuttlefish = cuttlefish_picker_random(current_frame, conf, cuttlefish)

    # _, init_frame = cap.read()
    # tracked_conf, tracked_cuttlefish = (0.73389, np.array([824.5, 798.5, 203/2, 151/2]))

    # Selected Cuttlefish
    tracked_cuttlefish[2:] /= 2
    print(tracked_conf, tracked_cuttlefish)

    # Create video output
    if args.save_video:
        outvid = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, img_size)

    # Create initial position
    ratio = img_size[0]/img_size[1]
    init_pos = np.array([[
        [tracked_cuttlefish[0], 0, 0, tracked_cuttlefish[1], 0, 0, tracked_cuttlefish[2], tracked_cuttlefish[3]],
        [(1-tracked_conf)*50*ratio, 0.1*ratio, 0.1*ratio, (1-tracked_conf)*50, 0.1, 0.1, (1-tracked_conf), (1-tracked_conf)]
    ]])

    # Create covariance matrices for prediction and update model
    # Sequence 1
    # Q_motion = np.array([[5*ratio, 0.1*fps*ratio, 0.2*fps*ratio, 5, 0.1*fps, 0.2*fps, 0.2*ratio, 0.2]]) # cap2Dbb lbp dbl MLE
    
    # Sequence 2
    # Q_motion = np.array([[5*ratio, 0.1*fps*ratio, 0.2*fps*ratio, 5, 0.1*fps, 0.2*fps, 0.2*ratio, 0.2]]) # ppp2Dbb lbp dbs MLE
    # Q_motion = np.array([[10*ratio, 0.1*fps*ratio, 0.4*fps*ratio, 10, 0.1*fps, 0.4*fps, 0.1*ratio, 0.1]]) # cap2Dbb hog dbs MLE
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.1*ratio, 0.1]]) # ppp2Dbb hogcascade dbs MLE R0.05 128 64
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.1*ratio, 0.1]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 128 64
    # Q_motion = np.array([[8*ratio, 0, 0, 8, 0, 0, 0.4, 0.4]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025) nbp16 r2
    # Q_motion = np.array([[5, 0, 0, 5, 0, 0, 0.8, 0.8]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025) nbp16 r2
    # Q_motion = np.array([[2, 0, 0, 4, 0, 0, 0.4, 0.4]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025) nbp16 r2
    # Q_motion = np.array([[0.5, 0, 0, 0.5, 0, 0, 0.2, 0.2]]) # ppp2Dbb hogcascadelbp cos MLE R0.1 64 64 np100 (gamma 0.0125 0.025) nbp16 r2
    # Q_motion = np.array([[0.1, 0, 0, 0.1, 0, 0, 0.1, 0.1]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025) nbp16 r2
    # Q_motion = np.array([[4, 0, 0, 8, 0, 0, 0.4, 0.8]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np500 (gamma 0.0125 0.025) nbp16 r2
    # Q_motion = np.array([[8*ratio, 0, 0, 8, 0, 0, 0.8, 1.0]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025) nbp12 r2

    # Sequence 4
    # Q_motion = np.array([[20*ratio, 10*fps*ratio, 20*fps*ratio, 20, 10*fps, 20*fps, 0.8*ratio, 0.8]]) # cap2Dbb hog bdl MAP
    # Q_motion = np.array([[1*ratio, 0.25*fps*ratio, 0.5*fps*ratio, 1, 0.25*fps, 0.25*fps, 0*ratio, 0]]) # cap2Dfbb hog bds MLE
    # Q_motion = np.array([[1*ratio, 0.5*fps*ratio, 1*fps*ratio, 1, 0.5*fps, 1*fps, 0.2*ratio, 0.2]]) # cap2Dbb hog bds MLE
    # Q_motion = np.array([[5*ratio, 0.4*fps*ratio, 0.8*fps*ratio, 5, 0.4*fps, 0.8*fps, 0.1*ratio, 0.1]]) # cap2Dbb hog bds MLE
    # Q_motion = np.array([[0.1*ratio, 0*fps*ratio, 0*fps*ratio, 0.1, 0*fps, 0*fps, 0.2*ratio, 0.2]]) # ppp2Dbb hog bds MLE
    # Q_motion = np.array([[1*ratio, 0*fps*ratio, 0*fps*ratio, 1, 0*fps, 0*fps, 1*ratio, 1]]) # ppp2Dbb hog bds MLE gaussianblur
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.1*ratio, 0.1]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 128 np100 (gamma 0.0125 0.025)
    # Q_motion = np.array([[8*ratio, 0, 0, 8, 0, 0, 0.2, 0.4]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 128 np100 (gamma 0.0125 0.025)
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.1*ratio, 0.1]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 128 np100
    # Q_motion = np.array([[10*ratio, 0, 0, 10, 0, 0, 0.2*ratio, 0.2]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025)
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.5*ratio, 0.5]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np500 (gamma 0.0125 0.025)
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.6, 0.9]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np500 (gamma 0.0125 0.025)
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.6, 0.9]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np500 (gamma 0.025 0.05)
    # Q_motion = np.array([[5*ratio, 0, 0, 5, 0, 0, 0.7, 1.0]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np500 (gamma 0.025 0.05)
    # Q_motion = np.array([[5, 0, 0, 5, 0, 0, 0.05, 0.05]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.025 0.05)
    # Q_motion = np.array([[5, 0, 0, 5, 0, 0, 1, 1]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np500 (gamma 0.025 0.05)
    # Q_motion = np.array([[2*ratio, 0, 0, 2, 0, 0, 0.2, 0.2]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np100 (gamma 0.0125 0.025) 2*w h/2
    # Q_motion = np.array([[4, 0, 0, 2, 0, 0, 0.4, 0.1]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np1000 (gamma 0.0125 0.025)
    Q_motion = np.array([[2*ratio, 0, 0, 2, 0, 0, 0.2, 0.2]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np1000 (gamma 0.0125 0.025)
    # Q_motion = np.array([[2, 0, 0, 2, 0, 0, 0.2, 0.2]]) # ppp2Dbb hogcascadelbp cos MLE R0.05 64 64 np1000 (gamma 0.025 0.05)
    
    # R = np.array([[0.1]])
    # R = np.array([[0.05]])
    R = np.array([[0.05, 0.2]])
    # R = np.array([[25]])

    # Set size of the descriptor and slicer
    desc_size = args.desc_size if args.desc_size!=None else (2*int(tracked_cuttlefish[2]), 2*int(tracked_cuttlefish[3]))

    # init functions and classes
    if args.descriptor == 'hog' or args.descriptor == 'hogcolor' or args.descriptor == 'hogcascade':
        descriptor = descriptor_dict[args.descriptor](desc_size, freezeSize=(args.desc_size!=None))
    elif args.descriptor == 'lbp':
        descriptor = descriptor_dict[args.descriptor](args.lbp_nbpoints, args.lbp_radius)
    elif args.descriptor == 'hogcascadelbp':
        descriptor = descriptor_dict[args.descriptor](desc_size, args.lbp_nbpoints, args.lbp_radius)
    else:
        descriptor = descriptor_dict[args.descriptor](args.nb_features)

    similarity = similarity_dict[args.similarity]
    resampling = resample_dict[args.resampling]
    slicer = slicer_dict[args.slicer](desc_size, np.copy(current_frame), freezeSize=(args.desc_size!=None))
    particle_struct = particle_dict[args.particle]

    # Initialize Particle filter
    particle_filter =  ParticleFilter(
        N, particle_struct, 1, 
        init_pos, np.copy(current_frame),
        args.alpha, Q_motion, R, 
        slicer, descriptor, similarity, resampling,
        seed)

    # Show Initial particles
    output_frame = draw_output_frame(np.copy(current_frame), particle_filter.mu)
    output_frame = draw_output_particles(output_frame, particle_filter.particles)
    output_frame = draw_output_mean_particule(output_frame, particle_filter.mu)
    # output_frame = draw_search_area(output_frame, particle_filter.mu, particle_filter.search_area)
    cv2.imshow('Track Cuttlefish', output_frame)
    cv2.waitKey(0)

    # Processing loop
    while not stop:

        # Read a frame
        ret, current_frame = cap.read()
        if not ret:
            print("Error : Couldn't read frame")
            quit()
        
        # Resize if video_size is not None
        if args.scale_factor != 1.:
            current_frame = cv2.resize(current_frame, (int(current_frame.shape[1]*args.scale_factor), int(current_frame.shape[0]*args.scale_factor)), interpolation=cv2.INTER_AREA)


        # Perform a pass of the particle filter
        particle_filter.forward(np.copy(current_frame), 1./fps, args.resample_factor)

        # Mean particle
        print(particle_filter.mu)
        print()

        # Draw Bbox and particles on the frame
        output_frame = draw_output_frame(np.copy(current_frame), particle_filter.mu)
        output_frame = draw_output_particles(output_frame, particle_filter.particles)
        output_frame = draw_output_mean_particule(output_frame, particle_filter.mu)
        # output_frame = draw_search_area(output_frame, particle_filter.mu, particle_filter.search_area)
        cv2.imshow('Track Cuttlefish', output_frame)
        # cv2.waitKey(0)

        # Write the frame into the video file
        if args.save_video:
            outvid.write(output_frame)
            np.savetxt('bboxSave.out', particle_filter.mu)

        # Press Q to stop the particle filter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True

    # Release the video capture and video write objects
    cap.release()
    if args.save_video:
        outvid.release()
    
    # Closes all the windows
    cv2.destroyAllWindows()
