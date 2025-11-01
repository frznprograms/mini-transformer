from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from loguru import logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.models.eval import ModelEvaluator
from src.models.model import MiniTransformer
from src.utils.helpers import set_device, set_seeds, load_data_splits
from src.utils.decorators import timed_execution

import copy
import json
from sklearn.model_selection import ParameterGrid


@dataclass
class ModelTrainer:
    config: dict
    device: str = "auto"
    seed: int = 42
    # config_path: str = "src/configs/all_configs.yaml"
    # device: str = "auto"
    # seed: int = 42

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

    @timed_execution
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

                if early_stop and self.loss_history and step % tol_steps == 0:
                    # implement early stopping
                    loss_after_tol_steps = (
                        self.loss_history[step - tol_steps]
                        - self.loss_history[step - 1]
                    )
                    if loss_after_tol_steps < tol:
                        logger.warning(
                            f"Stopping training at step {step} as loss has not been reduced by the set tolerance level of {tol} in {tol_steps} steps. Loss was only reduced by {loss_after_tol_steps:.4f} instead."
                        )
                        return {}

            avg_loss = total_loss / n
            self.avg_loss_history.append(avg_loss)
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}] | Average Loss: {avg_loss:.4f}"
            )
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


## Grid search manager
class GridSearchManager:
    def __init__(self, config_path: str, seed: int = 42, device: str = "auto"):
        self.config_path = config_path
        self.seed = seed
        self.device = device

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load grid search config: {e}")
            raise

        assert config is not None, f"Config file is empty or invalid: {config_path}"

        self.base_config = config["base_config"]
        self.grid_params = config["grid_search_params"]
        self.run_results = []

        logger.info("GridSearchManager initialized")

    ## function to run grid search
    def run(self, train_dataset, val_dataset):
        ## combinations of parameters
        all_param_combinations = list(ParameterGrid(self.grid_params))
        total_runs = len(all_param_combinations)
        logger.info(f"Starting grid search. Total experiments: {total_runs}")

        for i, params in enumerate(all_param_combinations):
            logger.info(f"---Starting run {i+1}/{total_runs}")
            current_config = copy.deepcopy(self.base_config)
            run_name = current_config["train"]["experiment_name"]

            for key, value in params.items():
                if key in current_config["model"]:
                    current_config["model"][key] = value
                elif key in current_config["train"]:
                    current_config["train"][key] = value
                run_name += f"_{key}{value}"  ## (?)

            current_config["train"]["experiment_name"] = run_name

            logger.info(f"Params: {json.dumps(params)}")

            try:
                trainer = ModelTrainer(
                    config=current_config, device=self.device, seed=self.seed
                )

                metrics = trainer.train(
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    plot_loss=False,
                )

                ## logging results
                self.run_results.append(
                    {
                        "run_name": run_name,
                        "params": params,
                        "final_val_loss": metrics["loss"],
                        "final_val_accuracy": metrics["accuracy"],
                    }
                )

                logger.success(
                    f"--- Finished Run {i+1}/{total_runs}. "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Accuracy: {metrics['accuracy']:.4f} ---"
                )

            except Exception as e:
                logger.error(f"Run {run_name} failed: {e}")
                self.run_results.append(
                    {
                        "run_name": run_name,
                        "params": params,
                        "final_val_loss": float("inf"),
                        "final_val_accuracy": 0.0,
                        "error": str(e),
                    }
                )

            self.report_results()

    def report_results(self):
        if not self.run_results:
            logger.warning("No results to report.")
            return
        logger.info("==================================================")
        logger.info("GRID SEARCH COMPLETE")

        best_loss_run = min(self.run_results, key=lambda x: x["final_val_loss"])
        best_acc_run = max(self.run_results, key=lambda x: x["final_val_accuracy"])

        logger.info("\n--- Best run by Loss ---")
        logger.info(f"Run: {best_loss_run['run_name']}")
        logger.info(f"Params: {best_loss_run['params']}")
        logger.info(f"Validation Loss: {best_loss_run['final_val_loss']:.4f}")
        logger.info(f"Validation Acc: {best_loss_run['final_val_accuracy']:.4f}")

        logger.info("\n--- Best run by accuracy ---")
        logger.info(f"Run: {best_acc_run['run_name']}")
        logger.info(f"Params: {best_acc_run['params']}")
        logger.info(f"Validation loss: {best_acc_run['final_val_loss']:.4f}")
        logger.info(f"Validation acc: {best_acc_run['final_val_accuracy']:.4f}")

        results_file = "grid_search_results.json"
        with open(results_file, "w") as f:
            json.dump(self.run_results, f, indent=2)
        logger.info(f"All run results saved to {results_file}")


# if __name__ == "__main__":
# mt = ModelTrainer()
# print(f"Model Configs: {mt.model_configs}")
# print(f"Train Configs: {mt.train_configs}")
# print(f"Device: {mt.device}")
# print(f"Optimizer: {mt.optimizer}")
# print(f"Criterion/Loss Function: {mt.criterion}")
# print(f"Model: {mt.model}")

# train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
# mt = ModelTrainer(
#     device="cpu", config_path="src/configs/all_configs.yaml", seed=42
# )

# mt.train(
#     train_dataset=train,
#     val_dataset=val,
#     plot_loss=False,
#     early_stop=True,
# tol=1.0,  # too high, just to test early stopping
# tol_steps=100,
# )
# mt.plot_loss()

## test block: checks if grid search works
if __name__ == "__main__":
    logger.info("Loading data for Grid Search Test...")
    train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
    logger.info("Loaded training and validation datasets.")

    config_file_path = "src/configs/test_grid_configs.yaml"

    search_manager = GridSearchManager(
        config_path=config_file_path, device="auto", seed=42
    )

    logger.info(f"Starting Grid Search test from {config_file_path}")
    search_manager.run(train, val)

    logger.success("Grid Search test complete")
