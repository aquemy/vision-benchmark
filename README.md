# CPU inference benchmark for object detection

## Model & Dataset

The selected model is [YOLOv7](https://github.com/WongKinYiu/yolov7) from the implemention proposed in PyTorch.   
The pre-trained model has been trained on MS COCO.

Modification to the algorithm:
1. I removed all the other capabilities of the algorithm (tracking, OCR, stream/video, segmentation) to keep only the object detection.
2. I removed all the training part to keep only the inference part.
3. I removed the GPU specific code and pinned the device to CPU wherever possible.

The selected dataset is a tiny subset of ImageNet available on Kaggle and called [imagenet-mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).


## Object Detection Pipeline


## Benchrmark Framework

