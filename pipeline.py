from pipelines.yolo_object_detector import YOLOPipeline
from utils.detections import draw
import json
import cv2
import torch 

from utils.datasets import load_dataset

dataset_name = 'imagenet-mini'
dataset = load_dataset(dataset_name)
imgs = dataset[:5]

torch.set_num_threads(20)

def inference_pipeline_seq(imgs, verbose: bool = False):
    """
        1. Load images
        2. Load the model
        3. Inference one after one
    """

    pipeline = YOLOPipeline()
    pipeline.load()

    for i, img in enumerate(imgs):
        detections = pipeline.detect(img)
        detected_image = draw(img, detections)
        cv2.imwrite(f'detected_image_{i}.jpg', detected_image)
        if verbose:
            print(json.dumps(detections, indent=4))


def inference_pipeline(imgs, verbose: bool = False):
    """
        1. Load images
        2. Load the model
        3. Batch inference
        4. Save all data at the end
    """

    pipeline = YOLOPipeline()
    pipeline.load()

    batch_detections = pipeline.detect_batch(imgs)
    for i, detections in enumerate(batch_detections):
        detected_image = draw(imgs[i], detections)
        cv2.imwrite(f'detected_image_{i}.jpg', detected_image)
        if verbose:
            print(json.dumps(detections, indent=4))

inference_pipeline(imgs)

