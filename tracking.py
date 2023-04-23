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
    parser.add_argument('--particle', type=str, default='cap2Dbb',
                        choices=['cap2Dfbb', 'cap2Dbb'],
                        help='which particle structure to use')
    parser.add_argument('--descriptor', type=str, default='hog',
                        choices=['brisk', 'sift', 'hog', 'orb', 'color'],
                        help='which descriptor to use')
    parser.add_argument('--similarity', type=str, default='bd',
                        choices=['bd'],
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

    # Video args
    parser.add_argument('--save-video', action='store_true', help='Save a video')
    parser.add_argument('--save-path', type=str, default='results', help='Path to save the video')


    return parser.parse_args()

def cuttlefish_picker(init_frame, conf, cuttlefish):
    error = (1-conf) * np.sqrt(((init_frame.shape[1]//2)-cuttlefish[:, 0])**2 + ((init_frame.shape[0]//2)-cuttlefish[:, 1])**2)
    index = np.argmin(error)
    return (conf[index], cuttlefish[index])

def draw_output_frame(frame : np.array, 
                    estimate_particle : np.array,
                    color=(0,255,0)):

    output_image = cv2.rectangle(frame, 
                        (int(estimate_particle[0] - estimate_particle[6]//2), int(estimate_particle[3] - estimate_particle[7]//2)),
                        (int(estimate_particle[0] + estimate_particle[6]//2), int(estimate_particle[3] + estimate_particle[7]//2)),
                        color,
                        2)
                    
    return output_image
    
    
def draw_output_particles(frame : np.array, 
                    particles : np.array,
                    color=(0,0,255)):

    for particle in particles[:, 0]:
        output_image = cv2.circle(frame, (int(particle[0]), int(particle[3])), radius=1, color=color, thickness=1)
    
    return output_image

def draw_output_mean_particule(frame : np.array, 
                    mean_particles : np.array,
                    color=(255,0,0)):

    output_image = cv2.circle(frame, (int(mean_particles[0]), int(mean_particles[3])), radius=3, color=color, thickness=3)
    
    return output_image

if __name__ == "__main__":
    # get params
    args = get_opts()

    # init params
    N = abs(args.N)
    stop = False
    model = Model(
        weights=args.weights, device=args.device, img_size=args.img_size,
        conf_thres=args.conf_thres, iou_thres=args.iou_thres)
    descriptor = descriptor_dict[args.descriptor]
    similarity = similarity_dict[args.similarity]
    resampling = resample_dict[args.resampling]
    slicer = args.slicer
    particle_struct = particle_dict[args.particle]

    # Load video
    cap = cv2.VideoCapture(args.filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Select a cuttlefish to track
    conf = []
    while not len(conf):
        # Read first frame
        ret, init_frame = cap.read()
        if not ret:
            print("Error : Couldn't read frame")
            quit()

        # Save image dimensions
        img_size = (init_frame.shape[1], init_frame.shape[1])

        # Run YOLOV7 to get bounding boxes
        conf, cuttlefish = model.detect(init_frame)

        if(len(conf)):
            tracked_conf, tracked_cuttlefish = cuttlefish_picker(init_frame, conf, cuttlefish)

    # _, init_frame = cap.read()
    # tracked_conf, tracked_cuttlefish = (0.73389, np.array([824.5, 798.5, 203, 151]))
    print(tracked_conf, tracked_cuttlefish)

    # Create video output
    if args.save_video:
        outvid = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (init_frame.shape[1], init_frame.shape[0]))

    init_pos = np.array([[
        [tracked_cuttlefish[0], 0, 0, tracked_cuttlefish[1], 0, 0, tracked_cuttlefish[2], tracked_cuttlefish[3]],
        [(1-tracked_conf), 0.1, 0.01, (1-tracked_conf), 0.1, 0.01, (1-tracked_conf), (1-tracked_conf)]
    ]])

    Q_motion = np.array([[10, 0.25, 0.025, 10, 0.25, 0.025, 10, 8]])
    R = np.array([[.2]])

    # Init Particle filter
    particle_filter =  ParticleFilter(N, particle_struct, 1, init_pos, init_frame, Q_motion, R, slicer, descriptor, similarity, resampling)

    print(particle_filter.mu)
    # Loop
    while not stop:
        ret, current_frame = cap.read() # Read next frame

        if not ret:
            print("Error : Couldn't read frame")
            quit()

        particle_filter.forward(current_frame, 1/fps) # New pass

        print(particle_filter.mu)
        print()
        output_frame = draw_output_frame(current_frame, particle_filter.mu)
        output_frame = draw_output_particles(output_frame, particle_filter.particles)
        output_frame = draw_output_mean_particule(output_frame, particle_filter.mu)
        cv2.imshow('Track Cuttlefish', output_frame)

        # Write the frame into the file 'output.avi'
        if args.save_video:
            outvid.write(output_frame)

        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop = True

    # End Loop

    # When everything done, release the video capture and video write objects
    cap.release()
    outvid.release()
    
    # Closes all the frames
    cv2.destroyAllWindows()
