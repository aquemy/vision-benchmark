import warnings
warnings.filterwarnings('ignore')
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.models import load_model
from utils.detections import Detections
from utils.datasets import letterbox
import numpy as np
import torch
import yaml
from functools import partial
import torch
import time

layer_time_dict = {}

def take_time_pre(layer_name, module, input):
    if layer_name not in layer_time_dict:
        layer_time_dict[layer_name] = []
    layer_time_dict[layer_name].append(time.time())

def take_time(layer_name, module, input, output):
    layer_time_dict[layer_name][-1] =  time.time() - layer_time_dict[layer_name][-1]


class YOLOPipeline:
    """ Pipeline for inference with YOLO.

    """
    def __init__(self, conf_thres: float = 0.25, iou_thres: float = 0.45, img_size: int = 640):
        self.settings = {
            'conf_thres': conf_thres,
            'iou_thres': iou_thres,
            'img_size': img_size
        }
        self._timeit = False
        self.forward_time = {
            'total': []
        }
        self.preprocess_time = {
            'total': []
        }
        self.postprocess_time = {
            'total': [],
            'NMS': [],
            'detection': []
        }


    def load(self, weights_path = 'coco.weights', classes='coco.yaml'):
        with torch.no_grad():
            self.model = load_model(weights_path).to("cpu")
            stride = int(self.model.stride.max())
            self.imgsz = check_img_size(self.settings['img_size'], s=stride)
            self.classes = yaml.load(open(classes), Loader=yaml.SafeLoader)['classes']

    def register_hooks(self):
        # Register function for every layer
        for m in self.model.modules():
            for layer in m.children():
                layer.register_forward_pre_hook(partial(take_time_pre, layer) )
                layer.register_forward_hook(partial(take_time, layer))
        self._timeit = True # Activate timer for pre and post-processing steps

    def preprocess_batch(self, images):
        # TODO: vectorize properly or threadpool
        if self._timeit:
            self.preprocess_time['total'] = time.time()
        batch = [img.copy() for img in images]
        batch = [letterbox(img, self.imgsz, auto=self.imgsz != 1280)[0] for img in batch]
        batch = [img[:, :, ::-1].transpose(2, 0, 1) for img in batch]
        batch = [np.ascontiguousarray(img) for img in batch]
        batch = [torch.from_numpy(img) for img in batch]
        batch = [img.float() / 255.0 for img in batch]
        unsqueeze = lambda x: x.unsqueeze(0) if x.ndimension() == 3 else x
        batch = [unsqueeze(img) for img in batch]
        if self._timeit:
            self.preprocess_time['total'] = time.time() - self.preprocess_time['total']

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
            if self._timeit:
                self.postprocess_time['total'].append(time.time())
            detections[i] = self.postprocess_image(preds[i], im0s[i], images[i])
            if self._timeit:
                self.postprocess_time['total'][-1] = time.time() - self.postprocess_time['total'][-1]
        return detections
    
    def postprocess_image(self, pred, im0, img):
        if self._timeit:
            self.postprocess_time['NMS'].append(time.time())
        pred = non_max_suppression(pred, self.settings['conf_thres'], self.settings['iou_thres'])
        if self._timeit:
            self.postprocess_time['NMS'][-1] = time.time() - self.postprocess_time['NMS'][-1]

        if self._timeit:
            self.postprocess_time['detection'].append(time.time())
        raw_detection = np.empty((0,6), float)
        for det in pred:
            if len(det) > 0:
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    raw_detection = np.concatenate((raw_detection, [[int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), round(float(conf), 2), int(cls)]]))
        detections = Detections(raw_detection, self.classes).to_dict()
        if self._timeit:
            self.postprocess_time['detection'][-1] = time.time() - self.postprocess_time['detection'][-1]
        return detections


    def detect_batch(self, images):
        with torch.no_grad():
            # TODO: vectorize properly or threadpool
            forward = self.model
            if self._timeit:
                def _timeit(img):
                    self.forward_time['total'].append(time.time())
                    r = self.model(img)
                    self.forward_time['total'][-1] = time.time() - self.forward_time['total'][-1]
                    return r
                forward = _timeit

            im0, images = self.preprocess_batch(images)
            preds = [forward(img)[0] for img in images]
            detections = self.postprocess_batch(preds, im0, images)
            return detections


    def detect(self, img):
        with torch.no_grad():
            im0, img = self.preprocess_image(img)
            pred = self.model(img)[0]
            detections = self.postprocess_image(pred, im0, img)
            return detections
        

    