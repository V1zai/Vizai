from clearml.automation import PipelineController

pipe = PipelineController(
    name="YOLOv8 Full Pipeline",
    project="Vizai",
    version="1.0"
)

pipe.add_step(
    name="Train YOLOv8",
    base_task_project="Vizai",
    base_task_name="YOLOv8 Training Pipeline", 
    parameter_override={
        "General/epochs": 50,
        "General/device": "cuda",   
    }
)

pipe.start()
