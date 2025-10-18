import torch
import torch.nn as nn

from dataclasses import dataclass
from src.models.model import MiniTransformer
from src.utils.helpers import set_device
from loguru import logger

import yaml


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

    def train(self):
        # set model to training mode
        self.model.train()
        epochs = self.train_configs["epochs"]

        logger.info("Starting Training...")
        for epoch in range(epochs):
            pass
