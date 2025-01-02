import os

import torch
import lightning as L
from lightning import seed_everything
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar
)

from utils import (
    get_args, 
    save_args,
    get_dataset,
    get_encoder_args,
    Encoder,
    Classifier
    )

def main():
    num_gpus = torch.cuda.device_count()
    data_dir = os.path.join("..", "data")
    arg_path = os.path.join("configs", "finetune.yaml")
    args = get_args(arg_path)
    seed_everything(args["seed"], workers=True)

    logger = TensorBoardLogger("finetune-runs", name=args["backbone"], version=args["experiment_num"])
    log_dir = os.path.join("finetune-runs", args["backbone"], f"version_{logger.version}")
    os.makedirs(log_dir, exist_ok=True)
    save_args(args, log_dir)

    save_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "finetune", f"version_{logger.version}")
    os.makedirs(save_dir, exist_ok=True)

    pbar = TQDMProgressBar(leave=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="highest-accuracy",
        monitor="Validation/Accuracy",
        mode="max",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        enable_version_counter=False
    )

    train_dataset = get_dataset(data_dir, split="train", apply_augmentation=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size = args["batch_size"],
        shuffle=True,
        num_workers=os.cpu_count() // 4,
        persistent_workers=True
    )

    val_dataset = get_dataset(data_dir, split="validation", apply_augmentation=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size = args["batch_size"],
        shuffle=False,
        num_workers=os.cpu_count() // 4,
        persistent_workers=True
    )

    run_dir = os.path.join("..", "src", "pre-train-runs", args["backbone"], f"version_{args['experiment_num']}", "run-config.yaml")
    ckpt_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "pre-train", f"version_{args['experiment_num']}", "min-loss.ckpt")

    ckpt = torch.load(ckpt_dir, map_location=torch.device("cpu"))["state_dict"]
    encoder_args = get_encoder_args(run_dir)
    
    encoder = Encoder(**encoder_args)
    student_params = {k: params for k, params in ckpt.items() if "student." in k}
    student_params = {k.replace("student.", ""): params for k, params in student_params.items()}

    encoder.load_state_dict(student_params)
    embedding_dim = encoder.mlp[0].in_features
    encoder = encoder.encoder

    strategy = "ddp" if num_gpus > 1 else "auto"
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    classifier = Classifier(
        encoder=encoder,
        num_classes=args["num_classes"],
        embedding_dim=embedding_dim,
        learning_rate=args["lr"],
        eta_min=args["eta_min"],
        weight_decay=args["weight_decay"]
    )

    trainer = L.Trainer(
        logger=logger,
        devices=-1,
        strategy=strategy,
        accelerator="auto", 
        deterministic=True,
        precision=precision,
        num_sanity_val_steps=0,
        max_epochs=args["epochs"], 
        callbacks=[checkpoint_callback, pbar]
    )

    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()