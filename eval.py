import os
import cv2
import argparse
import numpy as np
import xml.etree.ElementTree as ET

def bbox_iou_yolo(box1, box2, eps=1e-7):
    # Returns the IoU of box1 to box2.

    # Get the coordinates of bounding boxes
    b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
    b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = np.maximum(0, np.minimum(box1[2], b2_x2) - np.maximum(box1[0], b2_x1)) * \
            np.maximum(0, np.minimum(box1[3], b2_y2) - np.maximum(box1[1], b2_y1))

    # Union Area
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1] + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou

def bbox_iou_pf(box1, box2, eps=1e-7):
    # Returns the IoU of box1 to box2.

    # Get the coordinates of bounding boxes
    b2_x1, b2_x2 = box2[0] - box2[6], box2[0] + box2[6]
    b2_y1, b2_y2 = box2[3] - box2[7], box2[3] + box2[7]

    # Intersection area
    inter = np.maximum(0, np.minimum(box1[2], b2_x2) - np.maximum(box1[0], b2_x1)) * \
            np.maximum(0, np.minimum(box1[3], b2_y2) - np.maximum(box1[1], b2_y1))

    # Union Area
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1] + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou

def get_opts():
    parser = argparse.ArgumentParser()
    
    # Path to Bbox data
    parser.add_argument('--filepath', type=str, required=True, help='filepath of the Bbox data')
    
    # YOLO
    parser.add_argument('--yolo', action='store_true', help='Use if you compare the GT with a result from yolo')

    return parser.parse_args()

if __name__ == "__main__":
    # get params
    args = get_opts()

    GT = './results/GT'

    filenames = next(os.walk(GT), (None, None, []))[2]  # [] if no file

    if args.yolo:
        boxes2 = next(os.walk(args.filepath), (None, None, []))[2]
    else:
        boxes2 = np.loadtxt(args.filepath)

    iou = []

    cap = cv2.VideoCapture('./test/Cuttlefish-seq1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    outvid = cv2.VideoWriter('eval.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (int(width), int(height)))

    for file, box2 in zip(filenames, boxes2):

        ret, current_frame = cap.read()
        if not ret:
            print("Error : Couldn't read frame")
            quit()

        mytree = ET.parse(GT + '/' + file)
        myroot = mytree.getroot()
        box1 = np.array([
            int(myroot.find('object').find('bndbox').find('xmin').text),
            int(myroot.find('object').find('bndbox').find('ymin').text),
            int(myroot.find('object').find('bndbox').find('xmax').text),
            int(myroot.find('object').find('bndbox').find('ymax').text),
        ])

        if args.yolo:
            box2 = np.loadtxt(args.filepath + box2)[1:]
            iou.append(bbox_iou_yolo(box1, box2))
            current_frame = cv2.rectangle(
                current_frame, 
                (int(box2[0] - box2[2] / 2), int(box2[1] - box2[3] / 2)), 
                (int(box2[0] + box2[2] / 2), int(box2[1] + box2[3] / 2)), 
                (0, 0, 255), thickness=2)
        else:
            iou.append(bbox_iou_pf(box1, box2))
            current_frame = cv2.rectangle(
                current_frame, 
                (int(box2[0] - box2[6]), int(box2[3] - box2[7])), 
                (int(box2[0] + box2[6]), int(box2[3] + box2[7])),  
                (0, 0, 255), thickness=2)
        
        current_frame = cv2.rectangle(
            current_frame, 
            (box1[0], box1[1]), 
            (box1[2], box1[3]), 
            (0, 255, 0), thickness=2)
        
        current_frame = cv2.putText(current_frame, f'{iou[-1]}', (box1[0]-10, box1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Evaluation', current_frame)
        outvid.write(current_frame)
        cv2.waitKey(int(1000*(1/fps)))
    
    print(np.mean(np.array(iou)))
    np.savetxt('eval.out', np.array(iou))
    cap.release()
    outvid.release()
    cv2.destroyAllWindows()


