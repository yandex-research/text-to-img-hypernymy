"""Classifying the generated images."""

from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models


class Classifier:
    """A class for classifying diffusion-generated images."""

    def __init__(self, model_name: str, batch_size: int, gpu_id: int):
        self.model_name = model_name
        self.model = getattr(torchvision.models, model_name)(
            weights="IMAGENET1K_V1").to(f"cuda:{gpu_id}")
        self.model.eval()
        self.transform = torchvision.models.get_model_weights(
            model_name).IMAGENET1K_V1.transforms()
        self.batch_size = batch_size
        self.gpu_id = gpu_id

    @torch.no_grad()
    def compute_logits(self, images_path: str, logits_path: str):
        """
        Computes all logits for images in the `images_path` folder and saves
        them to `logits_path`.
        """

        logits_list_dict = defaultdict(list)

        dataset = ImageFolder(images_path, transform=self.transform)
        classes_list, _ = dataset.find_classes(images_path)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=4)

        torch.set_num_threads(16)

        for images_batch, classes_batch in tqdm.tqdm(dataloader):
            logits_batch = self.model(
                images_batch.to(f"cuda:{self.gpu_id}")).cpu().numpy()

            for logits, class_id in zip(logits_batch, classes_batch):
                logits_list_dict[classes_list[class_id]].append(logits)

        logits_dict = {k: np.stack(v) for k, v in logits_list_dict.items()}

        save_path = Path(logits_path)
        save_path.mkdir(parents=True, exist_ok=True)
        save_path = save_path / f"{self.model_name}_logits.npz"

        np.savez(save_path, **logits_dict)
