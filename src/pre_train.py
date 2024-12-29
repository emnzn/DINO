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
    get_base_lr,
    get_dataset,
    get_param_groups,
    cosine_scheduling,
    get_teacher_temperatures,
    DINO,
    Encoder
    )

def main():
    num_gpus = torch.cuda.device_count()
    data_dir = os.path.join("..", "data")
    arg_path = os.path.join("configs", "pre-train.yaml")
    args = get_args(arg_path)
    seed_everything(args["seed"], workers=True)

    logger = TensorBoardLogger("pre-train-runs", name=args["backbone"], version=args["experiment_num"])
    log_dir = os.path.join("pre-train-runs", args["backbone"], f"version_{logger.version}")
    os.makedirs(log_dir, exist_ok=True)
    save_args(args, log_dir)

    save_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "pre-train", f"version_{logger.version}")
    os.makedirs(save_dir, exist_ok=True)

    pbar = TQDMProgressBar(leave=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="min-loss",
        monitor="Loss",
        mode="min",
        save_weights_only=False,
        save_on_train_epoch_end=True,
        enable_version_counter=False
    )

    pretrain_dataset = get_dataset(
        cache_dir=data_dir,
        split="train",
        global_scale_min=args["global_scale_min"],
        global_scale_max=args["global_scale_max"],
        local_scale_min=args["local_scale_min"],
        local_scale_max=args["local_scale_max"],
        num_local_crops=args["num_local_crops"]
    )
    process_batch_size = args["batch_size"] if num_gpus == 0 else args["batch_size"] // num_gpus

    pretrain_loader = DataLoader(
        pretrain_dataset,
        shuffle=True,
        persistent_workers=True,
        num_workers=os.cpu_count() // 4,
        batch_size=process_batch_size
    )

    base_lr = get_base_lr(args["batch_size"])
    lr_schedule = cosine_scheduling(
        base_value=base_lr,
        final_value=args["final_lr"],
        epochs=args["epochs"],
        iters_per_epoch=len(pretrain_loader) // max(1, num_gpus),
        warmup_epochs=args["warmup_epochs_lr"],
        start_warmup_value=args["warmup_start_lr"]
        )
    
    student_temp = args["student_temp"]
    teacher_temp_schedule = get_teacher_temperatures(
        epochs=args["epochs"],
        num_warmup_epochs=args["warmup_teacher_epochs"],
        base_value=args["warmup_teacher_temp"],
        final_value=args["final_teacher_temp"]
    )

    weight_decay_schedule = cosine_scheduling(
        base_value=args["base_weight_decay"],
        final_value=args["final_weight_decay"],
        epochs=args["epochs"],
        iters_per_epoch=len(pretrain_loader) // max(1, num_gpus),
        warmup_epochs=0
    )

    teacher_momentum_schedule = cosine_scheduling(
        base_value=args["base_teacher_momentum"],
        final_value=args["final_teacher_momentum"],
        epochs=args["epochs"],
        iters_per_epoch=len(pretrain_loader) // max(1, num_gpus),
        warmup_epochs=0
    )

    encoder_params = ["backbone", "mlp_layers", "hidden_dim", "bottleneck_dim", "k_dim"]
    encoder_kwargs = {param: args[param] for param in encoder_params}

    student = Encoder(**encoder_kwargs)
    teacher = Encoder(**encoder_kwargs)

    center_momentum = args["center_momentum"]
    param_groups = get_param_groups(student)

    strategy = "ddp" if num_gpus > 1 else "auto"
    precision = "16-mixed" if torch.cuda.is_available() else "32"

    model = DINO(
        student=student,
        teacher=teacher,
        lr_schedule=lr_schedule,
        teacher_temp_schedule=teacher_temp_schedule,
        weight_decay_schedule=weight_decay_schedule,
        teacher_momentum_schedule=teacher_momentum_schedule,
        param_groups=param_groups,
        student_temp=student_temp,
        center_momentum=center_momentum,
        k_dim=args["k_dim"]
    )

    trainer = L.Trainer(
        logger=logger,
        devices=-1,
        strategy=strategy,
        accelerator="auto",
        precision=precision,
        max_epochs=args["epochs"],
        callbacks=[checkpoint_callback, pbar]
    )

    trainer.fit(model=model, train_dataloaders=pretrain_loader)
    
if __name__ == "__main__":
    main()