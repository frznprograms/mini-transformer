from typing import Optional

import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.segment import SegmentedCharDataset
from src.models.model_v3 import MiniTransformerV3
from src.utils.decorators import timed_execution
from src.utils.helpers import load_data_splits, set_device


class ModelEvaluator:
    def __init__(
        self,
        model_checkpoint_path: Optional[str],
        # model_config_path: str,
        model_config: dict,
        train_config: dict,
        device: str = "auto",
    ):
        self.model = None
        self.model_configs = model_config

        self.device = set_device(device)
        logger.info(f"Device has been set to {self.device}")

        self.model_configs = model_config
        self.train_configs = train_config
        if model_checkpoint_path is not None:
            self.model = MiniTransformerV3(**self.model_configs).to(self.device)
            self.model.load_state_dict(
                torch.load(
                    model_checkpoint_path,
                    map_location=torch.device(self.device),
                    weights_only=True,
                )
            )
        if self.model is not None:
            logger.success("Loaded model from checkpoint and configuration.")

    @timed_execution
    @logger.catch(message="Unable to complete model evaluation.", reraise=True)
    @torch.no_grad()
    def eval(self, dataset: SegmentedCharDataset) -> tuple[float, float]:
        self.model.eval()  # type: ignore
        batch_size = self.train_configs["batch_size"]

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        total_correct = 0
        total_tokens = 0
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        logger.info("Starting evaluation...")

        progress_bar = tqdm(dataloader, desc="Evaluating...", leave=False)
        for x, y in progress_bar:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)  # type: ignore
            B, T, C = logits.shape

            # compute loss
            loss = criterion(logits.view(B * T, C), y.view(B * T))  # type: ignore
            total_loss += loss.item()

            # predictions
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == y).sum().item()
            total_correct += correct
            total_tokens += y.numel()

            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        accuracy = total_correct / total_tokens

        logger.success(
            f"Evaluation complete. Accuracy: {accuracy:.4f}, Avg Loss: {avg_loss:.4f}"
        )
        return avg_loss, accuracy


if __name__ == "__main__":
    ## [testing] loading all_configs file
    config_path = "src/configs/base_configs.yaml"
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f)

    model_config = configs["model"]
    train_config = configs["train"]

    # with open("data/text8", "r") as f:
    #     full_text = f.read()

    # test_slice_text = full_text[:10000]
    # logger.info(f"Loaded test slice with {len(test_slice_text)} characters.")

    train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")

    logger.info("Loaded pre-segmented test dataset.")

    me = ModelEvaluator(
        model_checkpoint_path="checkpoints/base_model/checkpoint-FINAL/model.pt",
        model_config=model_config,
        train_config=train_config,
        device="cpu",
    )

    me.eval(dataset=test)

    # max_size = 50000
    # data = full_text[max_size : max_size + 10000]

    # train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
    # me = ModelEvaluator(
    #     model_checkpoint_path="checkpoints/base_model/checkpoint-FINAL/model.pt",
    #     model_config_path="src/configs/base_configs.yaml",
    #     device="cpu",
    # )
    # me.eval(dataset=test)
