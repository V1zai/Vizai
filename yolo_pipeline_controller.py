from clearml import PipelineController
from yolo_model_clearml import train_model

pipe = PipelineController(
    name='YOLOv8 Training Pipeline',
    project='Vizai',
    version='1.0'
)

# Step 1: Train model (calls your main training function)
pipe.add_function_step(
    name='train_model',
    function=train_model,
    function_kwargs={},
    cache_executed_step=False,
    execution_queue='default'
)
# Run the pipeline
pipe.start()