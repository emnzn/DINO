import os
from typing import List

import torch
import numpy as np
from tqdm import tqdm
import plotly.express as px
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from utils import (
    get_args,
    get_label_map,
    get_encoder_args,
    get_finetune_datasets,
    Encoder
)

@torch.no_grad()
def get_embeddings(
    data_loader: DataLoader,
    encoder: Encoder,
    device: str
    ) -> tuple[List[np.ndarray], List[str]]:

    encoder.eval()
    embeddings = []
    labels = []

    for img, label in tqdm(data_loader, desc="Encoding in progress"):
        img = img.to(device)
        label = label.cpu().numpy()

        embedding = encoder(img).squeeze().cpu().numpy()
        embeddings.extend(embedding)
        labels.extend(label)

    return embeddings, labels


def plot_embeddings(
    embeddings: List[np.ndarray],
    labels: List[str],
    dataset: str,
    save_dir: str,
    seed: int,
    ):

    label_map = get_label_map(dataset)

    embeddings = np.array(embeddings)
    labels = [label_map[i] for i in labels]

    tsne = TSNE(n_components=2, random_state=seed)
    projections = tsne.fit_transform(embeddings)

    fig = px.scatter(
        projections, x=0, y=1,
        color=labels, labels={"color": "label"}
    )
    fig.write_image(os.path.join(save_dir, f"{dataset}-embeddings.png"))


def main():
    data_dir = os.path.join("..", "data")
    arg_path = os.path.join("..", "src", "configs", "visualize-embedding.yaml")
    args = get_args(arg_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join("..", "assets", "embedding-visualization", args["backbone"], f"version_{args['experiment_num']}")
    os.makedirs(save_dir, exist_ok=True)

    _, val_dataset = get_finetune_datasets(
        dataset=args["dataset"],
        data_dir=data_dir
    )

    data_loader = DataLoader(
        val_dataset,
        batch_size=args["batch_size"],
        shuffle=False
    )

    run_dir = os.path.join("..", "src", "pre-train-runs", args["backbone"], f"version_{args['experiment_num']}", "run-config.yaml")
    ckpt_dir = os.path.join("..", "assets", "model-weights", args["backbone"], "pre-train", f"version_{args['experiment_num']}", "min-loss.ckpt")

    ckpt = torch.load(ckpt_dir, map_location=torch.device(device))["state_dict"]
    encoder_args = get_encoder_args(run_dir)

    encoder = Encoder(**encoder_args).to(device)

    student_params = {k: params for k, params in ckpt.items() if "student." in k}
    student_params = {k.replace("student.", ""): params for k, params in student_params.items()}

    encoder.load_state_dict(student_params)
    encoder = encoder.encoder
    embeddings, labels = get_embeddings(data_loader, encoder, device)

    plot_embeddings(embeddings, labels, args["dataset"], save_dir, args["seed"])


if __name__ == "__main__":
    main()