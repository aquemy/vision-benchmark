import warnings
warnings.filterwarnings('ignore')
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.models import load_model
from utils.detections import Detections
from utils.datasets import letterbox
import numpy as np
import torch
import yaml


class YOLOPipeline:
    def __init__(self, conf_thres: float = 0.25, iou_thres: float = 0.45, img_size: int = 640):
        self.settings = {
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'img_size': img_size
        }

    def load(self, weights_path = 'coco.weights', classes='coco.yaml'):
        with torch.no_grad():
            self.model = load_model(weights_path).to("cpu")
            stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.settings['img_size'], s=stride)
            self.classes = yaml.load(open(classes), Loader=yaml.SafeLoader)['classes']

    def preprocess_batch(self, images):
        # TODO: vectorize properly or threadpool
        batch = [img.copy() for img in images]
        batch = [letterbox(img, self.imgsz, auto=self.imgsz != 1280)[0] for img in batch]
        batch = [img[:, :, ::-1].transpose(2, 0, 1) for img in batch]
        batch = [np.ascontiguousarray(img) for img in batch]
        batch = [torch.from_numpy(img) for img in batch]
        batch = [img.float() / 255.0 for img in batch]
        unsqueeze = lambda x: x.unsqueeze(0) if x.ndimension() == 3 else x
        batch = [unsqueeze(img) for img in batch]

        return images, batch

    def preprocess_image(self, img):
        im0 = img.copy()
        img = letterbox(im0, self.imgsz, auto=self.imgsz != 1280)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img)
        img = img.float()
        img /= 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return im0, img
    
    def postprocess_batch(self, preds, im0s, images):
        # TODO: vectorize properly or threadpool
        detections = [None] * len(preds)
        for i in range(len(preds)):
            detections[i] = self.postprocess_image(preds[i], im0s[i], images[i])
        return detections
    
    def postprocess_image(self, pred, im0, img):
        pred = non_max_suppression(pred, self.settings['conf_thres'], self.settings['iou_thres'])
        raw_detection = np.empty((0,6), float)

        for det in pred:
            if len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    raw_detection = np.concatenate((raw_detection, [[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))
            
        return Detections(raw_detection, self.classes).to_dict()


    def detect_batch(self, images):
        with torch.no_grad():
            # TODO: vectorize properly or threadpool
            im0, images = self.preprocess_batch(images)
            preds = [self.model(img)[0] for img in images]
            detections = self.postprocess_batch(preds, im0, images)
            return detections


    def detect(self, img):
        with torch.no_grad():
            im0, img = self.preprocess_image(img)
            pred = self.model(img)[0]
            detections = self.postprocess_image(pred, im0, img)
            return detections
        

    