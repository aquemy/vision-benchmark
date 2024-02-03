import torch
import torch.nn as nn


def load_model(weights, inplace=True):
    from models.yolo import Detect

    checkpoint = torch.load(weights, map_location='cpu')  # load
    checkpoint = checkpoint['model'].float() # FP32 model
    model = checkpoint.fuse().eval()

    # Module compatibility updates
    for m in model.modules():
        t = type(m)
        if t is Detect:
            m.inplace = inplace  # torch 1.7.0 compatibility
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, 'anchor_grid')
                setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility
    return model
