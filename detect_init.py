
import cv2
import time
import torch
import random
import numpy as np

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device
from utils.datasets import letterbox

class Model(object):
    def __init__(self, weights: str ='weights/cuttlefish_best.pt', device: str ='0',
                img_size: int =640, conf_thres: float =0.5, iou_thres: float =0.45):
        # Initialize
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = select_device(device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)
        
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(img_size, s=self.stride)  # check img_size
        
        if self.half:
            self.model.half()  # to FP16
        

    def detect(self, image: np.ndarray):
        # Padded resize
        img = letterbox(image, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        
        # Run inference
        if self.device.type != 'cpu':
            self.model(img)  # run once
        old_img_w = old_img_h = self.img_size
        old_img_b = 1
        
        # Warmup
        if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                self.model(img)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = self.model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)[0]

        # Get names and colors
        names = ['Cuttlefish']
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        cuttlefish = []
        conf_list = []

        # Process detections
        # Rescale boxes from img_size to im0 size
        pred[:, :4] = scale_coords(img.shape[2:], pred[:, :4], image.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(pred):
            cuttlefish.append((xyxy2xywh(torch.tensor(xyxy).view(1, 4))).view(-1).tolist())  # normalized xywh
            conf_list.append(conf.item())

        #     label = f'{names[int(cls)]} {conf:.2f}'
        #     print(label)
        #     plot_one_box(xyxy, image, label=label, color=colors[int(cls)], line_thickness=1)
        
        # cv2.imshow("Results", image)
        # cv2.waitKey(1)
        return (np.array(conf_list), np.array(cuttlefish))
            
# model = Model(weights='weights/cuttlefish_best.pt', conf_thres=0.25)
# cap = cv2.VideoCapture("test/Cuttlefish-2.mp4")
# #image = cv2.imread("./utils/Squid_colors_2.jpg", cv2.IMREAD_COLOR)
# ret, image = cap.read()
# #image = cv2.resize(image, (640, 640))
# cuttlefish = model.detect(image)

# print(cuttlefish)