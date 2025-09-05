from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer, UniformIntegerParameterRange, UniformParameterRange

task = Task.init(project_name='Vizai', task_name='HPO YOLOv8 Training', task_type=Task.TaskTypes.optimizer)

args = task.connect({
    'base_train_task_id': 'your_base_train_task_id_here',
    'num_trials': 5,
    'time_limit_minutes': 30,
    'run_as_service': False,
    'test_queue': 'default',
    'num_epochs': 100,
    'batch_size': 8,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5
})

hpo = HyperParameterOptimizer(
    base_task_id=args['base_train_task_id'],
    hyper_parameters=[
        UniformIntegerParameterRange('num_epochs', min_value=50, max_value=args['num_epochs']),
        UniformIntegerParameterRange('batch_size', min_value=4, max_value=16),
        UniformParameterRange('learning_rate', min_value=1e-4, max_value=1e-2),
        UniformParameterRange('weight_decay', min_value=1e-6, max_value=1e-4)
    ],
    objective_metric_title='Validation Metrics',
    objective_metric_series='mAP50',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    total_max_jobs=args['num_trials'],
    execution_queue=args['test_queue'],
    parameter_override={
        'data_yaml_path': 'path/to/data.yaml'  # Set dataset path here
    }
)

hpo.start()
hpo.wait()
hpo.stop()
