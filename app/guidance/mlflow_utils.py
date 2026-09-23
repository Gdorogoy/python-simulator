"""Thin MLflow helpers shared by the training loops.
Uses MLflow's default local sqlite:///mlflow.db store unless MLFLOW_TRACKING_URI
is set. That backend's first-ever use races if several processes hit it at once
(each tries to create its schema) -- any future multi-process launcher should
create the experiment once, itself, before spawning workers, rather than
letting each worker's first start_run() call race to create it."""
import mlflow


def start_run(experiment_name: str, run_name: str = None, tags: dict = None):
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name, tags=tags)


def log_params_safe(params: dict):
    """mlflow.log_params requires string-able values under ~250 chars; stringify
    and truncate so odd param types (numpy arrays, None, objects) don't error out."""
    mlflow.log_params({k: str(v)[:250] for k, v in params.items()})


def log_metrics_safe(metrics: dict, step: int = None):
    """Drops None/non-numeric values (mlflow.log_metrics rejects both) and
    casts bools to 0/1 float."""
    numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v is not None}
    if numeric:
        mlflow.log_metrics(numeric, step=step)
