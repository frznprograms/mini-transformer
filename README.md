<h1 style="text-align: center;"> Mini Transformer for Next Word Prediction </h1>
<h2 style="text-align: center;"> Mini Transformer for Next Word Prediction </h2>

#### Authors

1. Shane Vivek Bharathan
2. Tan Shayne

#### NUS DSA4212-2510 Course Coordinator

1. Prof. Alexander Thiery

**Disclaimer**: We encourage use of our repository to support learning and deepening interests in machine learning/ deep learning. However, please do not reproduce our work as your own. Thanks, and we hope we like what you see here!

#### Setup Instructions

Install the `uv` package manager for your operating system using the following link:
<https://docs.astral.sh/uv/getting-started/installation/>

Then run:

```{bash}
uv venv
uv sync
```

This will install the needed dependencies for you automatically. Alternatively, if you prefer to use the `pip` package manager, simply initialise a new virtual environment, then run:

```{bash}
uv pip install -r requirements.txt
```

And if you would still prefer to use the original `pip` or `pip3` package manager (but really, why are you fighting me on this?), you may simply execute the following command:

```{bash}
pip install -r requirements.txt
```

#### Running Inference

You may also wish to simply deploy the model for direct inference without any training. For this purpose, we have left some of the better-performing model checkpoints under the checkpoints/ directory. The checkpoints have been given names based on the parameters they used, so they can be called as such in code to run inference. Please refer to the sample below to run inference using a model checkpoint titled "test-experiment-small-shane_batch_size32_d_ff256_d_model256_lr3e-05_n_heads4_n_layers4":

```{bash}
uv run -m src.models.inference
```

Of course, for my non-`uv` users, you may instead replace any `uv` commands for running scripts with the `python` command:

```{bash}
python run -m src.models.inference
```

### Reproducing Experiments

To reproduce our best-performing experiment, execute the following:

```{bash}
some_code wow
```

For those who wish to train a model using our architecture text8 dataset, we have made data preparation and model hyperparameters easily customisable, however you will need to configure this yourself, and we cannot guarantee high performance off-the-shelf. You may need to run several experiments with different configurations before you achieve strong performance.

Below is an example of how to create dataset splits in our implementation:

```{python}
prepare_dataset_somehow()
```

The next step after data preparation would be to set up the needed model hyperparameters. We make this easy using YAML files. You may save your desired configuration anywhere as long as it follows the correct format provided, but for cleanliness we recommend saving all configurations to the `configs/` folder under `src/`.

```{yaml}
model:
  param_1:
  param_2:

train:
  param_1:
  param_2:
```

Once done, you can move on to training the model on your prepared dataset:

```{python}
train_model()
```

#### Project Structure

Please refer to the diagram below for an overview of our project structure and what each component handles.
.
├── checkpoints
│   ├── experiment-small-shane
│   ├── test-experiment-small-shane_batch_size32_d_ff256_d_model256_lr3e-05_n_heads4_n_layers4
│   └── trial
├── data
│   ├── raw
│   │   └── text8
│   └── small
│   └── small_data.pt
├── logs
│   ├── error
│   │   ├── error_02_1216.log
│   │   └── error_14_1535.log
│   └── general
│   ├── all_02_1216.log
│   └── all_14_1535.log
├── main.py
├── output
│   ├── combined_results.json
│   ├── grid_search_results_shayne.json
│   └── grid_search_results.json
├── plots
│   ├── clusters_small.png
│   ├── heatmap_lr_dff_small.png
│   ├── heatmap_lr_dmodel_small.png
│   ├── heatmap_lr_nheads_small.png
│   ├── heatmap_lr_nlayers_small.png
│   └── parallel_plot_small.png
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   ├── analytics
│   │   └── analysis.py
│   ├── configs
│   │   ├── experiment_shayne.yaml
│   │   ├── experiments_shane.yaml
│   │   ├── logger_config.py
│   │   ├── test_configs.yaml
│   │   └── test_grid_configs.yaml
│   ├── datasets
│   │   ├── dataset.py
│   │   ├── prep.py
│   │   ├── segment.py
│   │   └── test.py
│   ├── models
│   │   ├── eval.py
│   │   ├── grid_search.py
│   │   ├── model.py
│   │   └── training.py
│   └── utils
│   ├── decorators.py
│   └── helpers.py
└── uv.lock

#### Conceptual Overview

#### References and Citations
