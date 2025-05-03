# clearml_controller.py
from clearml import PipelineController
from step1_yolo_dataset import upload_yolo_dataset
from step2_yolo_train import train_yolo_model

# Create the pipeline
pipe = PipelineController(
    name='YOLOv8 Pipeline',
    project='Vizai',
    version='1.0'
)

# Step 1: Upload Dataset
pipe.add_function_step(
    name='upload_dataset',
    function=upload_yolo_dataset,
    function_kwargs={},
    cache_executed_step=False,
    execution_queue='default'
)

# Step 2: Train Model
pipe.add_function_step(
    name='train_model',
    function=train_yolo_model,
    function_kwargs={},
    cache_executed_step=False,
    execution_queue='default',
    parents=['upload_dataset'] 
)

# Run the pipeline
pipe.start()
