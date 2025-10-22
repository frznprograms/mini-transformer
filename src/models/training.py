from dataclasses import dataclass

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.datasets.dataset import CharDataset
from src.models.model import MiniTransformer
from src.utils.helpers import set_device

from pathlib import Path

# TODO: implement early stopping


@dataclass
class ModelTrainer:
    config_path: str = "src/configs/base_configs.yaml"
    device: str = "auto"

    def __post_init__(self) -> None:
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
        logger.info(f"Device has been set to {self.device}")

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
        self.loss_history = []

    @logger.catch(message="Unable to complete model training.", reraise=True)
    def train(self, dataset: CharDataset, plot_loss: bool = False) -> None:
        self.model.train()
        num_epochs = self.train_configs["epochs"]
        batch_size = self.train_configs["batch_size"]

        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        n = len(dataloader)

        logger.info(f"Starting training for {num_epochs} epochs...")

        step = 0
        for epoch in range(num_epochs):
            total_loss = 0.0

            progress_bar = tqdm(
                dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
            )

            for x, y in progress_bar:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                B, T, C = logits.shape
                loss = self.criterion(logits.view(B * T, C), y.view(B * T))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                self.loss_history.append(loss.item())
                progress_bar.set_postfix(loss=loss.item())
                step += 1

                self._save_model_checkpoint(step=step, pbar=progress_bar)

            avg_loss = total_loss / n
            logger.info(
                f"Epoch [{epoch+1}/{num_epochs}] | Average Loss: {avg_loss:.4f}"
            )

        self._save_final_checkpoint()
        logger.success("Training complete.")
        if plot_loss:
            self.plot_loss()

    def _save_model_checkpoint(self, step: int, pbar):
        experiment_name = self.train_configs.get(
            "experiment_name", "unnamed_experiment"
        )
        checkpoint_dir = Path(f"checkpoints/{experiment_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        save_strategy = self.train_configs.get("save_strategy", None)
        save_steps = self.train_configs.get("save_steps", None)
        should_save_checkpoint = (
            save_strategy == "steps"
            and isinstance(save_steps, int)
            and step % save_steps == 0
        )

        if should_save_checkpoint:
            try:
                # create specific checkpoint folder
                step_dir = checkpoint_dir / f"checkpoint-{step}"
                step_dir.mkdir(parents=True, exist_ok=True)
                model_file = step_dir / "model.pt"

                torch.save(self.model.state_dict(), model_file)
                pbar.set_postfix_str(f"Saved at step {step}")

            except Exception as e:
                logger.warning(f"Unable to save model checkpoint at step {step}: {e}")

    def _save_final_checkpoint(self):
        experiment_name = self.train_configs.get(
            "experiment_name", "unnamed_experiment"
        )
        checkpoint_dir = Path(f"checkpoints/{experiment_name}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        try:
            final_dir = checkpoint_dir / "checkpoint-FINAL"
            final_dir.mkdir(parents=True, exist_ok=True)
            model_file = final_dir / "model.pt"

            torch.save(self.model.state_dict(), model_file)
            logger.success(f"Final model checkpoint saved at {model_file}")

        except Exception as e:
            logger.warning(f"Unable to save final model checkpoint: {e}")

    def plot_loss(self) -> None:
        plt.plot(self.loss_history, marker="o")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Cross Entropy Loss through time")
        plt.show()


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
    mt = ModelTrainer(device="cpu")

    mt.train(dataset=cd)
    mt.plot_loss()
