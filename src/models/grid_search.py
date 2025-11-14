import copy
import json

import yaml
from loguru import logger
from sklearn.model_selection import ParameterGrid

from src.configs.logger_config import LoggedProcess
from src.datasets.segment import SegmentedCharDataset
from src.models.training import ModelTrainer
from src.utils.decorators import timed_execution
from src.utils.helpers import load_data_splits


class GridSearchManager(LoggedProcess):
    def __init__(
        self,
        config_path: str,
        seed: int = 42,
        device: str = "auto",
        output_dir: str = "logs",
    ):
        super().__init__(output_dir=output_dir)
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

        logger.success("GridSearchManager initialized")

    @timed_execution
    def run(
        self,
        train_dataset: SegmentedCharDataset,
        val_dataset: SegmentedCharDataset,
        plot_loss: bool = False,
        early_stop: bool = True,
        tol: float = 0.005,
        tol_steps: int = 1000,
        patience: int = 2,
    ):
        ## combinations of parameters
        all_param_combinations = list(ParameterGrid(self.grid_params))
        total_runs = len(all_param_combinations)
        logger.info(f"Starting grid search. Total experiments: {total_runs}")

        for i, params in enumerate(all_param_combinations):
            logger.info(f"---Starting run {i + 1}/{total_runs}")
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
                    plot_loss=plot_loss,
                    early_stop=early_stop,
                    tol=tol,
                    tol_steps=tol_steps,
                    patience=patience,
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
                    f"--- Finished Run {i + 1}/{total_runs}. "
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

        results_file = "output/grid_search_results.json"
        with open(results_file, "w") as f:
            json.dump(self.run_results, f, indent=2)
        logger.info(f"All run results saved to {results_file}")


## test block: checks if grid search works
if __name__ == "__main__":
    logger.info("Loading data for Grid Search Test...")
    train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
    logger.success("Loaded training and validation datasets.")

    config_file_path = "src/configs/experiments_shane.yaml"
    # config_file_path = "src/configs/experiments_shayne.yaml"

    search_manager = GridSearchManager(
        config_path=config_file_path, device="auto", seed=42
    )

    logger.info(f"Starting Grid Search test from {config_file_path}")
    search_manager.run(train, val)

    logger.success("Grid Search test complete")
