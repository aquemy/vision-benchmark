from pipelines.yolo_object_detector import YOLOPipeline
from utils.detections import draw
from utils.postprocess import post_process_layer_timer, post_process_pipeline_steps_timer
import json
from pathlib import Path
import cv2
import torch 
from pandas import concat
from datetime import datetime
from utils.datasets import load_dataset
from pyJoules.energy_meter import measure_energy
from pyJoules.device.rapl_device import RaplPackageDomain, RaplCoreDomain
from pyJoules.handler.pandas_handler import PandasHandler
from torch.profiler import profile, record_function, ProfilerActivity
pandas_handler = PandasHandler()

torch.set_num_threads(10)

def inference_pipeline_seq(pipeline, imgs, output, verbose: bool = False):
    """
        1. Load images
        2. Load the model
        3. Inference one after one
    """
    for i, img in enumerate(imgs):
        detections = pipeline.detect(img)
        detected_image = draw(img, detections)
        cv2.imwrite(str(output / 'images' / f'image_{i}.jpg'), detected_image)
        if verbose:
            print(json.dumps(detections, indent=4))


def inference_pipeline(pipeline, imgs, output, verbose: bool = False):
    """
        1. Load images
        2. Load the model
        3. Batch inference
        4. Save all data at the end
    """
    batch_detections = pipeline.detect_batch(imgs)
    for i, detections in enumerate(batch_detections):
        detected_image = draw(imgs[i], detections)
        cv2.imwrite(str(output / 'images' / f'image_{i}.jpg'), detected_image)
        if verbose:
            print(json.dumps(detections, indent=4))

def _init_run():
    output = Path('./output')

    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    run_output = output / run_id
    run_output.mkdir(parents=True, exist_ok=True)
    (run_output / 'images').mkdir(parents=True, exist_ok=True)
    return run_output

if __name__ == "__main__":

    run_output = _init_run()

    # Benchmark configuration 
    run_config = {
        'img_count': 10,  # Number of images in the run
        'dataset_name': 'imagenet-mini'
    }

    metrics_config = {
        'layer_time': True,
        'energy': True
    }

    # Add the RAPL decorator for energy measurement
    inference_pipeline = measure_energy(
        handler=pandas_handler, 
        domains=[RaplPackageDomain(0), RaplCoreDomain(0)])(inference_pipeline)

    dataset_name = run_config['dataset_name']
    dataset = load_dataset(dataset_name)
    imgs = dataset[:run_config['img_count']]

    # Create the pipeline and load the model on CPU
    pipeline = YOLOPipeline()
    pipeline.load()

    # Register the hooks to time all layers and steps
    if metrics_config.get('layer_time', False):
        pipeline.register_hooks()

    ### RUN START
    
    # Use PyTorch profiler, notably because it does the warmup for us
    # We could also create a more specific schedule but for the exercise it will do
    with profile(activities=[ProfilerActivity.CPU], 
                 record_shapes=True,
                 with_stack=True) as prof:
        with record_function("YOLO Inference Pipeline"):
            inference_pipeline(pipeline, imgs, run_output)
    
    ### ENDS
            
    ### RUN POST-PROCESS
    print("\n#### CPU Time and Call Count:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(str((run_output / "trace.json").absolute()))

    if metrics_config.get('layer_time', False):
        from pipelines.yolo_object_detector import layer_time_dict
        layer_time_df = post_process_layer_timer(layer_time_dict)
        #print(layer_time_df) # Time for each of the 250 layers -> save directly to csv for further study / dashboard
        layer_time_df.to_csv((run_output / 'layer_timer.csv').absolute())
        print("\n#### Time spent per layer type:")
        print(layer_time_df.groupby('layer_type')['total'].sum().sort_values(ascending=False).to_string())

        print('\n#### Time spent per pipeline steps:')
        aggregated = post_process_pipeline_steps_timer(pipeline)
        aggregated.to_csv((run_output / 'steps_timer.csv').absolute())
        aggregated_steps = aggregated.T[[c for c in aggregated.T if 'total' in c]].T
        aggregated_steps['total %'] =  100 * aggregated_steps['total'] / aggregated_steps['total'].sum()
        print(aggregated_steps.drop(columns=['runs']).to_string())


    print('\n#### Total Energy (in uJ):')
    energy_df = pandas_handler.get_dataframe()
    energy_df.to_csv((run_output / 'energy.csv').absolute())
    print(energy_df)

