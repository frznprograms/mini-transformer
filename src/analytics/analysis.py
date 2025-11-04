import json
import time
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from typing import Any, Optional
from dataclasses import dataclass, field
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
        print("#### First 5 Rows ####")
        print(self.data_table.head())  # type: ignore
        print("\n")
        print("#### Columns ####")
        print(self.data_table.columns)  # type: ignore

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
        self,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        cols=[
            "params.d_model",
            "params.d_ff",
            "params.n_heads",
            "params.n_layers",
            "final_val_accuracy",
        ],
    ) -> None:
        df_viz = self.data_table.copy()  # type: ignore
        df_viz["final_val_accuracy"] = df_viz["final_val_accuracy"].round(3)
        df_viz["lr"] = df_viz["params.lr"].astype(str)

        plt.figure(figsize=(10, 6))
        parallel_coordinates(
            df_viz,
            class_column="params.lr",
            cols=[cols],
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

    def get_optimal_n_clusters(self):
        pass

    def plot_clusters(
        self,
        cols: list[str] = [
            "params.batch_size",
            "params.d_ff",
            "params.d_model",
            "params.lr",
            "params.n_heads",
            "params.n_layers",
        ],
        random_state: int = 42,
        save_plot: bool = False,
        save_path: Optional[str] = None,
        show_table: bool = True,
    ) -> pd.DataFrame:
        X = self.data_table[cols]  # type: ignore
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        logger.success("Scaled input successfully.")

        n = self.get_optimal_n_clusters()
        k_means = KMeans(n_clusters=n, random_state=random_state)
        logger.info(f"Initialised KMeans with n = {n} clusters.")

        df_copy = self.data_table.copy()  # type: ignore
        df_copy["cluster"] = k_means.fit_predict(X_scaled)
        logger.success("Clustered data successfully.")

        # Get centroids (in original scale)
        centroids = pd.DataFrame(
            scaler.inverse_transform(k_means.cluster_centers_),
            columns=cols,  # type: ignore
        )

        # Get average performance for each cluster
        cluster_perf = (
            df_copy.groupby("cluster")[["final_val_accuracy", "final_val_loss"]]
            .mean()
            .reset_index()
        )

        summary = centroids.join(cluster_perf.set_index("cluster"), how="left")
        summary["cluster"] = summary.index
        summary = summary.reset_index(drop=True)

        if show_table:
            print("\n#### Summary of Centroids and Performance ####")
            print(summary.round(4))

        scaled_cols = cols + ["final_val_accuracy"]
        scaled_df = summary.copy()
        scaled_df[scaled_cols] = MinMaxScaler().fit_transform(summary[scaled_cols])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Hyperparameters only
        sns.heatmap(
            scaled_df.set_index("cluster")[cols],
            annot=summary[cols].round(6),
            cmap="coolwarm",
            fmt=".1f",
            ax=axes[0],
        )
        axes[0].set_title("Hyperparameter Patterns per Cluster")
        axes[0].set_ylabel("Cluster")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha="right")

        # Performance metrics
        sns.heatmap(
            scaled_df.set_index("cluster")[["final_val_accuracy"]],
            annot=summary[["final_val_accuracy"]].round(3),
            cmap="YlGnBu",
            fmt=".3f",
            ax=axes[1],
        )
        axes[1].set_title("Average Validation Accuracy per Cluster")
        axes[1].set_ylabel("")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        if save_plot:
            if not save_path:
                timestamp = time.time()
                save_path = f"plots/cluster_heatmap_{timestamp}.png"
            try:
                plt.savefig(save_path, bbox_inches="tight")
                logger.success(f"Saved plot to {save_path}")
            except Exception as e:
                logger.warning("Unable to save plot; please check the path.")
                raise e

        plt.show()
        return summary

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
    # r.plot_heatmap(var_1="lr", var_2="d_model")
    r.plot_clusters()
