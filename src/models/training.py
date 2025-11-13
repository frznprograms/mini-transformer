from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.eval import ModelEvaluator
from src.models.model import MiniTransformer
from src.utils.helpers import set_device, set_seeds


@dataclass
class ModelTrainer:
    config: dict
    device: str = "auto"
    seed: int = 42

    def __post_init__(self) -> None:
        configs = self.config
        # ensure weight initialisation is reproducible
        set_seeds(self.seed)
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
        self.avg_loss_history = []
        self.avg_val_history = []
        self.val_accuracy = []

    @logger.catch(message="Unable to complete model training.", reraise=True)
    def train(
        self,
        train_dataset,
        val_dataset,
        plot_loss: bool = False,
        early_stop: bool = False,
        tol: float = 0.01,
        tol_steps: int = 1000,
    ) -> dict:
        self.model.train()
        num_epochs = self.train_configs["epochs"]
        batch_size = self.train_configs["batch_size"]

        dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        n = len(dataloader)

        logger.info(f"Starting training for {num_epochs} epochs...")

        step = 0
        for epoch in range(num_epochs):
            total_loss = 0.0
            progress_bar = tqdm(
                dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
            )
            # main training loop
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

                # implement early stopping for convergence
                if early_stop and self.loss_history and step % tol_steps == 0:
                    self._check_potential_convergence(
                        step=step, tol_steps=tol_steps, tol=tol
                    )

            avg_loss = total_loss / n
            self.avg_loss_history.append(avg_loss)
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] | Average Loss: {avg_loss:.4f}"
            )

            # implement early stopping for potential overfitting
            if early_stop:
                if epoch > 0:
                    # compare current and previous epoch
                    if self.avg_loss_history[-1] >= self.avg_loss_history[-2]:
                        logger.warning(
                            "Stopping training early as average epoch loss has either stagnated or worsened."
                        )
                    return {}

            # validation loop
            if val_dataset is not None:
                self.evaluator = ModelEvaluator(
                    model_checkpoint_path=None,  # can skip loading here
                    model_config=self.model_configs,
                    train_config=self.train_configs,
                    # model_config_path=self.config_path,
                    device=self.device,
                )
                self.evaluator.model = self.model
                avg_val_loss, accuracy = self.evaluator.eval(dataset=val_dataset)
                self.avg_val_history.append(avg_val_loss)
                self.val_accuracy.append(accuracy)

        self._save_final_checkpoint()
        logger.success("Training complete.")
        if plot_loss:
            self.plot_loss()

        # final metrics from final epoch
        final_loss = self.avg_val_history[-1] if self.avg_val_history else float("inf")
        final_acc = self.val_accuracy[-1] if self.val_accuracy else 0.0

        return {"loss": final_loss, "accuracy": final_acc}

    def _check_potential_convergence(
        self, step: int, tol_steps: int, tol: float
    ) -> Union[dict, None]:
        # implement early stopping for convergence
        loss_after_tol_steps = (
            self.loss_history[step - tol_steps] - self.loss_history[step - 1]
        )
        if loss_after_tol_steps < tol:
            logger.warning(
                f"Stopping training at step {step} as loss has not been reduced by the set tolerance level of {tol} in {tol_steps} steps. Loss was only reduced by {loss_after_tol_steps:.4f} instead."
            )
            return {}
        return None

    def _check_potential_overfitting(self, epoch: int) -> Union[dict, None]:
        if epoch > 0:
            # compare current and previous epoch
            if self.avg_loss_history[-1] >= self.avg_loss_history[-2]:
                logger.warning(
                    "Stopping training early as average epoch loss has either stagnated or worsened."
                )
            return {}
        return None

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
        plt.figure(figsize=(16, 10))

        # Step-wise training loss
        plt.subplot(3, 1, 1)
        plt.plot(self.loss_history, color="blue", linewidth=1)
        plt.title("Step-wise Training Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.6)

        # Average loss per epoch (train vs val)
        plt.subplot(3, 1, 2)
        plt.plot(self.avg_loss_history, label="Train Avg Loss", marker="o")
        if self.avg_val_history:
            plt.plot(self.avg_val_history, label="Val Avg Loss", marker="o")
        plt.title("Average Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        # Validation accuracy
        if self.val_accuracy:
            plt.subplot(3, 1, 3)
            plt.plot(
                self.val_accuracy,
                label="Validation Accuracy",
                color="green",
                marker="o",
            )
            plt.title("Validation Accuracy per Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend()

        plt.tight_layout()
        plt.show()


# if __name__ == "__main__":
#     mt = ModelTrainer()
#     # print(f"Model Configs: {mt.model_configs}")
#     # print(f"Train Configs: {mt.train_configs}")
#     # print(f"Device: {mt.device}")
#     # print(f"Optimizer: {mt.optimizer}")
#     # print(f"Criterion/Loss Function: {mt.criterion}")
#     # print(f"Model: {mt.model}")
#
#     train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
#     mt = ModelTrainer(device="cpu", config_path="src/configs/all_configs.yaml", seed=42)
#
#     mt.train(
#         train_dataset=train,
#         val_dataset=val,
#         plot_loss=False,
#         early_stop=True,
#         tol=1.0,  # too high, just to test early stopping
#         tol_steps=100,
#     )
#     mt.plot_loss()
