from pipelines.yolo_object_detector import YOLOPipeline
from utils.detections import draw
import json
import cv2
import torch 

from utils.datasets import load_dataset


torch.set_num_threads(20)

def inference_pipeline_seq(pipeline, imgs, verbose: bool = False):
    """
        1. Load images
        2. Load the model
        3. Inference one after one
    """
    for i, img in enumerate(imgs):
        detections = pipeline.detect(img)
        detected_image = draw(img, detections)
        cv2.imwrite(f'detected_image_{i}.jpg', detected_image)
        if verbose:
            print(json.dumps(detections, indent=4))


def inference_pipeline(pipeline, imgs, verbose: bool = False):
    """
        1. Load images
        2. Load the model
        3. Batch inference
        4. Save all data at the end
    """
    batch_detections = pipeline.detect_batch(imgs)
    for i, detections in enumerate(batch_detections):
        detected_image = draw(imgs[i], detections)
        cv2.imwrite(f'detected_image_{i}.jpg', detected_image)
        if verbose:
            print(json.dumps(detections, indent=4))


if __name__ == "__main__":

    metrics_config = {
        'layer_time': True
    }

    dataset_name = 'imagenet-mini'
    dataset = load_dataset(dataset_name)
    imgs = dataset[:5]

    pipeline = YOLOPipeline()
    pipeline.load()
    if metrics_config.get('layer_time', False):
        pipeline.register_hooks()

    inference_pipeline(pipeline, imgs)

    if metrics_config.get('layer_time', False):
        from pipelines.yolo_object_detector import layer_time_dict
        print(layer_time_dict)

