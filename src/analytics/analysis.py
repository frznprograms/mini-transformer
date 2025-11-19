import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from loguru import logger
from pandas.plotting import parallel_coordinates
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.decorators import timed_execution


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
        max_loss: Optional[float] = None,
        save_to_local: bool = False,
    ) -> list[dict[str, Any]]:
        best_results = []
        for entry in self.data:
            if entry["final_val_accuracy"] >= min_acc:
                if max_loss is not None:
                    if entry['final_val_loss'] <= max_loss:
                        best_results.append(entry)
                # if max_loss and entry["final_val_loss"] <= max_loss:
                #     best_results.append(entry)
                #     continue
                else:
                    best_results.append(entry)

        if save_to_local:
            timestamp = time.time()
            with open(f"output/best_results_{timestamp}", "w") as f:
                json.dump(best_results, f, indent=4)

        return best_results
    
    def rank_loss_acc(
            self,
            results: list[dict],
            w_acc: float = 0.5,
            w_loss: float = 0.5,
    ) -> list[dict]:
        """
        Ranks configurations obtained from get_best_result(min_acc, max_loss).
        Takes loss and accuracy into account by normalizing and adding them up.
        Range of final measure = [0, 1]
        """
        if not results:
            return []
        
        accuracies = [r['final_val_accuracy'] for r in results]
        losses = [r['final_val_loss'] for r in results]

        min_acc, max_loss = min(accuracies), max(losses) # worst scores among top k configurations
        max_acc, min_loss = max(accuracies), min(losses) # best scores among top k configurations

        acc_range = max_acc - min_acc if max_acc != min_acc else 1.0
        loss_range = max_loss - min_loss if max_loss != min_loss else 1.0

        scored_results = []

        for r in results:
            norm_acc = (r['final_val_accuracy'] - min_acc)/acc_range
            norm_loss = 1 - ((r['final_val_loss'] - min_loss)/loss_range)

            weighted_sum = (w_acc * norm_acc) + (w_loss * norm_loss)

            r_copy = r.copy()
            r_copy['_combined_score'] = weighted_sum
            scored_results.append(r_copy)
        ranked = sorted(scored_results, key=lambda x: x['_combined_score'], reverse=True)
        for r in ranked:
            del[r['_combined_score']]
        return ranked


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
        df_viz["params.lr"] = df_viz["params.lr"].astype(str)

        # scale numeric columns independently
        numeric_cols = [c for c in cols if c != "params.lr"]
        scaler = MinMaxScaler()
        df_viz[numeric_cols] = scaler.fit_transform(df_viz[numeric_cols])

        plt.figure(figsize=(10, 6))
        parallel_coordinates(
            df_viz,
            class_column="params.lr",
            cols=cols,
            color=["#4c72b0", "#55a868", "#c44e52"],
        )
        plt.title(
            "Parallel Coordinates — Effect of Hyperparameters on Validation Accuracy (Normalised)"
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

    @timed_execution
    def get_optimal_n_clusters(
        self, X_scaled: pd.DataFrame, min: int = 2, max=10, random_state: int = 42
    ) -> int:
        best_k = 0
        max_score = float("-inf")
        for i in range(min, max):
            k_means = KMeans(n_clusters=i, random_state=random_state).fit(X_scaled)
            score = silhouette_score(X=X_scaled, labels=k_means.labels_)
            if score > max_score:
                max_score = score
                best_k = i

        logger.info(
            f"Best k for this case is {best_k} with silhouette score {max_score}."
        )
        return best_k

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

        n = self.get_optimal_n_clusters(X_scaled=X_scaled)
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
    with open("output/combined_results.json", "r") as f:
        data = json.load(f)
    r = ResultsAnalyser(data=data)  # type: ignore

    # best_results = r.get_best_results(min_acc=0.55)
    # print(best_results)

    # r.plot_parallel_coordinates(save_plot=True, save_path="plots/parallel_plot_small")

    r.plot_heatmap(
        var_1="lr",
        var_2="d_model",
        save_plot=True,
        save_path="plots/heatmap_lr_dmodel_small",
    )
    r.plot_heatmap(
        var_1="lr", var_2="d_ff", save_plot=True, save_path="plots/heatmap_lr_dff_small"
    )
    r.plot_heatmap(
        var_1="lr",
        var_2="n_heads",
        save_plot=True,
        save_path="plots/heatmap_lr_nheads_small",
    )
    r.plot_heatmap(
        var_1="lr",
        var_2="n_layers",
        save_plot=True,
        save_path="plots/heatmap_lr_nlayers_small",
    )
    r.plot_clusters(save_plot=True, save_path="plots/clusters_small")
