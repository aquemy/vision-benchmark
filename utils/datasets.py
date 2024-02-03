import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
import glob
from pandas.core.common import flatten

DATASETS_PATH = Path('./datasets')

# TODO: in yaml file
datasets = {
    'imagenet-mini': 'imagenet-mini/val'
}

def load_dataset(dataset_name: str):
    # TODO read from yaml
    img_path = datasets[dataset_name]

    return BenchmarkDataset(str((DATASETS_PATH / img_path).absolute()))


class BenchmarkDataset(Dataset):
    """ Basic image loader for algorithm hardware benchmark purpose. 
    Does not retain labels as it is made to evaluate the algorithm on a problem but its hardware perf.

    Note(aquemy): the torchvision version of this YOLO implementation does not have torchvision.datasets.ImageFolder
    """
    def __init__(self, img_path):
        img_paths = []
        for data_path in glob.glob(img_path +  '/*'):
            img_paths.append(glob.glob(data_path + '/*'))
            
        self.image_paths  = list(flatten(img_paths))
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        images_filepath = self.image_paths[idx]
        if isinstance(images_filepath, str):
            image = cv2.imread(images_filepath)
            return image
        else:
            images = [cv2.imread(img) for img in images_filepath]
            return images

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)