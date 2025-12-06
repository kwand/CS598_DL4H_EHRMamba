import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import click
import torch
from pyhealth.datasets import MIMIC4Dataset, get_dataloader, split_by_patient, split_by_sample
from pyhealth.tasks import Readmission30DaysMIMIC4
from utils.trainer import Trainer

from models.mamba_mpy import Mamba as MambaMPY
from models.mamba2_mpy import Mamba2 as Mamba2MPY
from tasks.tasks import BinaryLengthOfStayPredictionMIMIC4, MortalityPrediction31DaysMIMIC4
from utils.optimization import build_linear_scheduler_with_warmup_and_decay

DATA_ROOT = "datasets/mimic-iv-2.2"
EHR_TABLES = ["patients", "admissions", "diagnoses_icd", "procedures_icd", "prescriptions"]
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 20
OUTPUT_DIR = Path("results")
SEED = 42

ModelBuilder = Callable[[Any], Any]

TASK_BUILDERS: Dict[str, Callable[[], Tuple[Any, Callable, str, int]]] = {
    "readmission_30_days": lambda: (
        Readmission30DaysMIMIC4(), # task
        split_by_sample, # split fn
        "cache/readmission_prediction", # cache dir
        4, # num workers
    ),
    "length_of_stay_prediction": lambda: (
        BinaryLengthOfStayPredictionMIMIC4(),
        split_by_sample,
        "cache/length_of_stay_prediction",
        1,
    ),
    "mortality_prediction": lambda: (
        MortalityPrediction31DaysMIMIC4(),
        split_by_patient,
        "cache/mortality_prediction",
        1,
    ),
}

MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "mamba_mpy": lambda ds: MambaMPY(dataset=ds, embedding_dim=128, num_layers=16, dropout=0.1),
    "mamba2_mpy": lambda ds: Mamba2MPY(dataset=ds, embedding_dim=128, num_layers=16, dropout=0.1),
}


def prepare_dataloaders(task_name: str, batch_size: int):
    if task_name not in TASK_BUILDERS:
        raise ValueError(f"Unsupported task '{task_name}'. Choose from {list(TASK_BUILDERS)}.")

    task, split_fn, cache_dir, num_workers = TASK_BUILDERS[task_name]()
    dataset = MIMIC4Dataset(
        ehr_root=DATA_ROOT,
        ehr_tables=EHR_TABLES,
        dev=False,
    )
    dataset_with_task = dataset.set_task(
        task=task,
        cache_dir=cache_dir,
        num_workers=num_workers,
    )
    train_dataset, val_dataset, test_dataset = split_fn(dataset_with_task, ratios=[0.85 * 0.9, 0.85 * 0.1, 0.15], seed=SEED)

    train_loader = get_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    return dataset_with_task, train_loader, val_loader, test_loader


def build_model(model_name: str, dataset) -> Any:
    if model_name not in MODEL_BUILDERS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(MODEL_BUILDERS)}.")
    return MODEL_BUILDERS[model_name](dataset)


def save_metrics(metrics: Dict[str, Any], model_name: str, task_name: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUT_DIR / f"{model_name}_{task_name}_{timestamp}.json"
    payload = {
        "model": model_name,
        "task": task_name,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return filename


@click.group()
def cli():
    """Train EHR models for different tasks."""


@cli.command()
@click.argument("model", type=click.Choice(list(MODEL_BUILDERS.keys()), case_sensitive=False))
@click.argument("task", type=click.Choice(list(TASK_BUILDERS.keys()), case_sensitive=False))
@click.option("--epochs", default=DEFAULT_EPOCHS, show_default=True, help="Number of training epochs.")
@click.option("--batch-size", default=DEFAULT_BATCH_SIZE, show_default=True, help="Batch size for dataloaders.")
@click.option("--device", default="cuda", show_default=True, help="Device passed to Trainer.")
def train(model: str, task: str, epochs: int, batch_size: int, device: str):
    """Train a specified MODEL on the given TASK."""
    model = model.lower()
    task = task.lower()

    click.echo(f"Preparing data for task: {task}")
    dataset_with_task, train_loader, val_loader, test_loader = prepare_dataloaders(task, batch_size)

    click.echo(f"Initializing model: {model}")
    model_instance = build_model(model, dataset_with_task)
    trainer = Trainer(model=model_instance, metrics=["roc_auc", "pr_auc", "f1"], device=device)

    total_steps = len(train_loader) * epochs
    click.echo(f"Starting training for {epochs} epochs ({total_steps} steps)...")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="roc_auc",
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": 5e-5},
        scheduler_class_or_fn=build_linear_scheduler_with_warmup_and_decay,
        scheduler_params={"n_steps": total_steps, "warmup_ratio": 0.1, "decay_ratio": 0.9},
    )

    click.echo("Evaluating on the test set...")
    metrics = trainer.evaluate(test_loader)
    metrics_path = save_metrics(metrics, model, task)
    click.echo(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    cli()
