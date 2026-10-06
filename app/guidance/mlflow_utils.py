"""Thin MLflow helpers. Store = MLFLOW_TRACKING_URI or sqlite:///mlflow.db (create experiments before forking workers)."""
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
