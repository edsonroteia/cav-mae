"""Wandb integration utilities for CAV-MAE/JEPA training."""

import wandb
import os


def init_wandb(args, model_name="cav-mae"):
    """Initialize wandb run with training config.

    Args:
        args: Argument namespace containing training config
        model_name: Name prefix for the run (e.g., "cav-mae", "cav-jepa")

    Returns:
        wandb.Run object if wandb is enabled, None otherwise
    """
    if not getattr(args, 'use_wandb', False):
        return None

    # Convert args to config dict
    config = vars(args).copy()
    # Remove non-serializable items
    for key in list(config.keys()):
        if callable(config[key]):
            del config[key]

    # Generate run name if not provided
    run_name = getattr(args, 'wandb_run_name', None)
    if run_name is None:
        run_name = f"{model_name}-{args.dataset}-lr{args.lr}-ep{args.n_epochs}"

    # Get project name
    project = getattr(args, 'wandb_project', 'cav-mae')

    # Initialize wandb
    run = wandb.init(
        project=project,
        name=run_name,
        config=config,
        dir=args.exp_dir,
        resume="allow"
    )

    print(f"Wandb initialized: {run.url}")
    return run


def log_train_step(metrics, step, use_wandb=True):
    """Log per-step training metrics.

    Args:
        metrics: Dict of metric name -> value
        step: Global training step
        use_wandb: Whether wandb is enabled
    """
    if use_wandb and wandb.run is not None:
        wandb.log(metrics, step=step)


def log_epoch_metrics(train_metrics, val_metrics, epoch, lr, use_wandb=True):
    """Log per-epoch metrics including validation.

    Args:
        train_metrics: Dict of training metrics
        val_metrics: Dict of validation metrics
        epoch: Current epoch number
        lr: Current learning rate
        use_wandb: Whether wandb is enabled
    """
    if use_wandb and wandb.run is not None:
        all_metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        all_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        all_metrics["epoch"] = epoch
        all_metrics["lr"] = lr
        wandb.log(all_metrics)


def log_best_model(model_path, use_wandb=True):
    """Log best model checkpoint as artifact.

    Args:
        model_path: Path to the model checkpoint
        use_wandb: Whether wandb is enabled
    """
    if use_wandb and wandb.run is not None:
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Best model checkpoint"
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)


def finish_wandb():
    """Clean up wandb run."""
    if wandb.run is not None:
        wandb.finish()
