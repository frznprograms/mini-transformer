import json
import time
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResultsAnalyser:
    """
    Class to analyse results from multiple experiments. Class is
    frozen to ensure data for analysis cannot be accidentally overriden
    between results comparisons.
    Args:
        data (list[dict[str, Any]): json format results from experiments
    """

    data: list[dict[str, Any]]
    data_table: Optional[pd.DataFrame] = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "data_table", self.convert_to_table())

    def view_data_table(self) -> None:
        print(self.data_table.head())  # type: ignore

    def get_best_results(
        self,
        min_acc: float,
        min_loss: Optional[float] = None,
        save_to_local: bool = False,
    ) -> list[dict[str, Any]]:
        best_results = []
        for entry in self.data:
            if entry["final_val_accuracy"] >= min_acc:
                if min_loss and entry["final_val_loss"] >= min_loss:
                    best_results.append(entry)
                    continue
                else:
                    best_results.append(entry)

        if save_to_local:
            timestamp = time.time()
            with open(f"output/best_results_{timestamp}", "w") as f:
                json.dump(best_results, f, indent=4)

        return best_results

    def plot_parallel_coordinates(
        self, save_plot: bool = False, save_path: Optional[str] = None
    ) -> None:
        df_viz = self.data_table.copy()  # type: ignore
        df_viz["final_val_accuracy"] = df_viz["final_val_accuracy"].round(3)
        df_viz["lr"] = df_viz["params.lr"].astype(str)

        plt.figure(figsize=(10, 6))
        parallel_coordinates(
            df_viz,
            class_column="params.lr",
            cols=[
                "params.d_model",
                "params.d_ff",
                "params.n_heads",
                "params.n_layers",
                "final_val_accuracy",
            ],
            color=["#4c72b0", "#55a868", "#c44e52"],
        )
        plt.title(
            "Parallel Coordinates — effect of hyperparameters on validation accuracy"
        )
        if save_plot:
            if not save_path:
                timestamp = time.time()
                save_path = f"plots/parallel_coordinates_plot_at_{timestamp}.png"
            try:
                plt.savefig(save_path)
            except Exception as e:
                logger.warning(
                    "Unable to save plot, please ensure path has been specified correctly"
                )
                raise e

        plt.show()

    def plot_heatmap(
        self,
        var_1: str,
        var_2: str,
        save_plot: bool = False,
        save_path: Optional[str] = None,
    ) -> None:
        pivot = self.data_table.pivot_table(  # type: ignore
            values="final_val_accuracy",
            index=f"params.{var_1}",
            columns=f"params.{var_2}",
            aggfunc="mean",
        )
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, cmap="viridis")
        plt.title(
            f"Heatmap comparing Model Performance for different combinations of {var_1} and {var_2}"
        )
        if save_plot:
            if not save_path:
                timestamp = time.time()
                save_path = f"plots/heatmap_{var_1}_vs_{var_2}_at_{timestamp}.png"
            try:
                plt.savefig(save_path)
            except Exception as e:
                logger.warning(
                    "Unable to save plot, please ensure path has been specified correctly"
                )
                raise e

        plt.show()

    def plot_clusters(self):
        pass

    def convert_to_table(self) -> pd.DataFrame:
        df = pd.json_normalize(self.data)
        return df


if __name__ == "__main__":
    with open("output/grid_search_results.json", "r") as f:
        data = json.load(f)
    r = ResultsAnalyser(data=data)
    # r.view_data_table()
    # best_results = r.get_best_results(min_acc=0.55)
    # print(best_results)
    # r.plot_parallel_coordinates()
    r.plot_heatmap(var_1="lr", var_2="d_model")
