import torch
import torch.nn as nn

from tqdm.auto import tqdm
import yaml
from loguru import logger
from src.datasets.dataset import CharDataset
from src.models.model import MiniTransformer
from src.utils.helpers import set_device

from torch.utils.data import DataLoader
from typing import Optional


# TODO: find out where to set seed


class ModelEvaluator:
    def __init__(
        self,
        model_checkpoint_path: Optional[str],
        model_config_path: str,
        device: str = "auto",
    ):
        configs = None
        self.model = None
        try:
            with open(model_config_path, "r") as f:
                configs = yaml.safe_load(f)
        except Exception as e:
            logger.error(
                "Unable to load Configurations. Please check that the file path is correct, and see the error below: "
            )
            raise e

        self.device = set_device(device)
        logger.info(f"Device has been set to {self.device}")

        self.model_configs = configs["model"]
        self.train_configs = configs["train"]
        if model_checkpoint_path is not None:
            self.model = MiniTransformer(**self.model_configs).to(self.device)
            self.model.load_state_dict(
                torch.load(model_checkpoint_path, weights_only=True)
            )
        if self.model is not None:
            logger.success("Loaded model from checkpoint and configuration.")

    @torch.no_grad()
    def eval(self, dataset: CharDataset) -> tuple[float, float]:
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
    with open("data/text8", "r") as f:
        full_text = f.read()

    max_size = 50000
    data = full_text[max_size : max_size + 10000]

    cd = CharDataset(text=data, context_size=10)
    me = ModelEvaluator(
        model_checkpoint_path="checkpoints/base_model/checkpoint-FINAL/model.pt",
        model_config_path="src/configs/base_configs.yaml",
        device="cpu",
    )
    me.eval(dataset=cd)
