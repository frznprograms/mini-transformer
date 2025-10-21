from dataclasses import dataclass

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.dataset import CharDataset
from src.models.model import MiniTransformer
from src.utils.helpers import set_device


@dataclass
class ModelTrainer:
    config_path: str = "src/configs/base_configs.yaml"
    device: str = "auto"

    def __post_init__(self):
        configs = None
        try:
            with open(self.config_path, "r") as f:
                configs = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                "Unable to load Configurations. Please check that the file path is correct, and see the error below: "
            )
            raise e

        self.device = set_device(self.device)
        self.model_configs = configs["model"]
        self.train_configs = configs["train"]
        self.model = MiniTransformer(**self.model_configs).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_configs["lr"]
        )
        self.criterion = nn.CrossEntropyLoss()
        logger.success(
            "Model, Optimizer, Loss, and Configurations loaded successfully."
        )

    def train(self, dataset: CharDataset):
        # set model to training mode
        self.model.train()
        num_epochs = self.train_configs["epochs"]
        batch_size = self.train_configs["batch_size"]

        n = len(dataset)
        num_steps = num_epochs * (n // batch_size)

        logger.info("Starting Training...")
        pbar = tqdm(num_steps)
        for epoch in range(num_epochs):
            total_loss = 0.0

            for x, y in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                x, y = x.to(self.device), y.to(self.device)

                # forward
                logits = self.model(x)
                B, T, C = logits.shape

                loss = self.criterion(logits.view(B * T, C), y.view(B * T))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pbar.update(1)

            avg_loss = total_loss / n
            print(f"Epoch: {epoch} | Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # mt = ModelTrainer()
    # print(f"Model Configs: {mt.model_configs}")
    # print(f"Train Configs: {mt.train_configs}")
    # print(f"Device: {mt.device}")
    # print(f"Optimizer: {mt.optimizer}")
    # print(f"Criterion/Loss Function: {mt.criterion}")
    # print(f"Model: {mt.model}")

    with open("data/text8", "r") as f:
        full_text = f.read()

    max_size = 1000
    data = full_text[:max_size]

    cd = CharDataset(text=data, context_size=10)
    mt = ModelTrainer()

    mt.train(dataset=cd)
