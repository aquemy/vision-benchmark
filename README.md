# CPU inference benchmark for object detection

## Model & Dataset

The selected model is [YOLOv7](https://github.com/WongKinYiu/yolov7) from the implemention proposed in PyTorch.   
The pre-trained model has been trained on MS COCO.

Modification to the algorithm:
1. I removed all the other capabilities of the algorithm (tracking, OCR, stream/video, segmentation) to keep only the object detection.
2. I removed all the training part to keep only the inference part.
3. I removed the GPU specific code and pinned the device to CPU wherever possible.
4. I had to create an equivalent to `torchvision.datasets.ImageFolder` as it does not exist in the torchvision version required by this YOLO implementation.
5. I had to write the batch inference method because the implementation offered only sequential inference. However, I did not implement neither a proper vectorization nor concurrent processing.

The selected dataset is a tiny subset of ImageNet available on Kaggle and called [imagenet-mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000).


## Object Detection Pipeline

Structure of the project:
```
/models  # YOLO implementation
/pipelines
    |- yolo_object_detector.py  # Pipeline object responsible for (pre,post)processing, forward and timing
/utils
    |- dataset.py     # General loader for img as PyTorch Dataset + letterbox
    |- detections.py  # Bounding box
    |- general.py     # Preprocessing and postprocessing methods
    |- models.py      # Model loader
benchmark.py          # Execution of the pipeline, metric measurements and postprocessing
```



## Benchmark Framework

The benchmark consists in three steps:

1. Loading a configurable number of images and loading the model.
2. Executing the pipeline over the batch of images. In particular, the mode is warmed up on CPU using PyTorch Profiler.
3. Collecting, processing and exporting the metrics in the `outputs/<run_id>/` folder. 


## Metrics & Artefacts

All the artefacts are saved in `outputs/<run_id>/`. The pictures with the bouding boxes are in `outputs/<run_id>/images`.

### CPU Time:   

Because we are interested in inference on CPU only, a system clock should be OK.

1. For layers, the measurement is done by registering forward hooks on the model hosted by the Pipeline object. Tere 251 layers in the model so the full CSV is exported in `outputs/<run_id>/layer_timer.csv` which can be further explored. However, we compute and display the aggregated time per layer as this tells us where to look for optimization:
```
#### Time spent per layer type:
layer_type
Conv           3.753335
Conv2d         3.022981
BatchNorm2d    0.486310
Detect         0.478222
MaxPool2d      0.304365
SP             0.219757
LeakyReLU      0.219396
Concat         0.175794
MP             0.093428
Upsample       0.044974
```
2. For the pre-processing and post-processing, the measurement is done manually in the Pipeline object. Because I did not have time to implement a proper vectorization and/or threapool for the batch methods, it is OK to measure it that way. I save also the time for the Non-maximum Suppression and the Bounding Box calculation as substep of the postprocessing. The CSV is exported in `outputs/<run_id>/steps_timer.csv`. But as it turns out preprocessing and postprocessing steps are neglectible, the summary only displays `preprocessing`, `forward` and `postprocessing`:
```
#### Time spent per pipeline steps:
                      total count       avg       std       min       max    total %
total_preprocess   0.101738    10  0.010174       0.0  0.010174  0.010174   2.023684
total_forward      4.852746    10  0.485275  0.059022  0.400288  0.578981  96.526841
total_postprocess   0.07287    10  0.007287  0.003465  0.001867  0.012716   1.449475
```
3. As PyTorch Profiler is used, we also export the trace in `outputs/<run_id>/trace.json` that can be explored using `chrome://tracing`:
![Trace](assets/trace_export.png)

### Energy:

Another important metric to measure, especially for hardware comparison, is the energy consumption.

I used PyJoules which uses RAPL technology to estimate the CPU and core consumption of the overall benchmark. Because we have a specified number of images during the benchmark, we can easily convert it to inference/W or FLOP/W or MAC/W.

The values are exported in `outputs/<run_id>/energy.csv`.

Example of output:
```
      timestamp                 tag  duration   package_0      core_0
0  1.706986e+09  inference_pipeline  5.212322  79850138.0  32277140.0
```

### RAM

Probably a bit less important in the context of CPU but can still give some importnat hints no the model's behavior.

```
mprof run python benchmark.py 
mprof plot -o RAM_usage.png --backend agg
```
![RAM usage](./assets/RAM_usage.png)

### The missing ones

- **model MACs / FLOPs:** I was not able to get the FLOPS using PAPI because this laptop I have a custom kernel without support for libpfm4 and I did not want to take the risk to recompile my kernel:
```
(.venv) aquemy@ws:~/projects/benchmark/vision-benchmark$ papi_component_avail 
Available components and hardware information.
--------------------------------------------------------------------------------
PAPI version             : 6.0.0.0
Operating system         : Linux 6.5.0-10022-tuxedo
Vendor string and code   : GenuineIntel (1, 0x1)
Model string and code    : 13th Gen Intel(R) Core(TM) i7-13700H (186, 0xba)
CPU revision             : 2.000000
CPUID                    : Family/Model/Stepping 6/186/2, 0x06/0xba/0x02
CPU Max MHz              : 4800
CPU Min MHz              : 400
Total cores              : 20
SMT threads per core     : 2
Cores per socket         : 10
Sockets                  : 1
Cores per NUMA region    : 20
NUMA regions             : 1
Running in a VM          : no
Number Hardware Counters : 0
Max Multiplex Counters   : 384
Fast counter read (rdpmc): yes
--------------------------------------------------------------------------------

Compiled-in components:
Name:   perf_event              Linux perf_event CPU counters
   \-> Disabled: Unknown libpfm4 related error
Name:   perf_event_uncore       Linux perf_event CPU uncore and northbridge
   \-> Disabled: No uncore PMUs or events found

Active components:
```

- A fine grain CPU profiling using VTune. I did not have time to integrate it into the project but I've done a couple of exploration which shows the limtation of the GIL of the vanilla Python implementation.
![Alt text](assets/vtune_1.png)
![Alt text](assets/vtune_2.png)

# Adding a dataset

The pipeline accepts any size of images. The Dataset loader allows for images of different sizes as well.

To add a new dataset, modify the ```utils/dataset.py``` and add an entry to the dictionary `datasets`:
```
datasets = {
    'imagenet-mini': 'imagenet-mini/val'
    'my_dataset': '<path/to/images>
}
```
(It should be a YAML file to make it easier...)

# TODO

- Better configuration of the benchmark suite via YAML files (dataset, config, metrics, etc).
- Integrate `mprof` directly in the benchmark rather than a separate command.
- Integrating VTune directly in the benchmark.
- Improving the benchmark using Cython, CPU pinning, OpenVINO or ONNX.
- Streamlit or Tensorboard to present all the results.