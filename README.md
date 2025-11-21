<div align="center">src="title.png", alt="title pic", width="350"></div>
<h3 align="center"> Mini Transformer for Next Word Prediction </h3>

#### Description

This project was completed as part of our submission for DSA4212-2510. In this project, we explore some of the intricacies of deep learning, with a particular focus on model design, training methodology, and optimisation methods.

We conducted a series of experiments in an attempt to maximise the performance of out model on **next-character prediction** tasks. Our particular model architecture is similar to NanoGPT(see a brief summary of NanoGPT by Shawn Chumbar here: <https://medium.com/@shawn.chumbar/understanding-nanogpt-a-deep-dive-into-transformer-architecture-implementation-9a7167b7d58c>), but we have taken pains to make the training loop, model design and experiments as accesible and extensible as possible to allow others to play with it.

#### Results

| Model Name/Checkpoint             | Configurations                     | Accuracy |
| --------------------------------- | ---------------------------------- | -------- |
| checkpoints/final/checkpoint-5000 | See src/configs/final_configs.yaml | 0.5536   |

#### Authors

1. Shane Vivek Bharathan
2. Tan Shayne

#### NUS DSA4212-2510 Course Coordinator

1. Prof. Alexander Thiery

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

**Note: All our imports are absolute to avoid strange import errors in Python, and all scripts are run as modules in testing using the `-m` flag. We recommend you do the same.**

#### Running Inference

To use our best model and reproduce the accuracy result, simply run this in command line:

```{bash}
uv run -m src.models.eval
```

### Reproducing Experiments

To reproduce our best-performing experiment, execute the following:

```{bash}
from src.models.grid_search import GridSearchManager
from src.utils.helpers import load_data_splits

logger.info("Loading data for Grid Search Test...")
train, val, test, encoded = load_data_splits(path="data/small/small_data.pt")
logger.success("Loaded training and validation datasets.")

config_file_path = "src/configs/experiments_shane.yaml"
# config_file_path = "src/configs/experiments_shayne.yaml" can be used too

search_manager = GridSearchManager(
    config_path=config_file_path, device="auto", seed=42
)

logger.info(f"Starting Grid Search test from {config_file_path}")
search_manager.run(train, val)

logger.success("Grid Search test complete")
```

Do note that the above code will run a full **grid search** on the stated hyper-parameters and may time a while to run, depending on your hardware. If you do not have a GPU on your device, we recommend using Google Colab, Kaggle or other free compute resources. Please also do not run a full grid search unless absolutely needed.

For those who wish to train a model using our architecture text8 dataset, we have made data preparation and model hyper-parameters easily customisable, however you will need to configure this yourself, and we cannot guarantee high performance off-the-shelf. You may need to run several experiments with different configurations before you achieve strong performance.

Below is an example of how to create dataset splits in our implementation:

```{python}
from src.utils.helpers import save_text8_splits


with open("data/raw/text8", "r") as f:
    full_text = f.read()

max_size = 100000 # set  your preferred size here
data = full_text[:max_size]

save_text8_splits(
    text=data,
    path="data/small/small_data.pt", # change path as needed
    ratios=(0.8, 0.1, 0.1),
    segment_len=4096,
    context_size=128,
    seed=1,
)
```

This will create the dataset of the desired size and save it to the directory you specify in the `save_path` parameter. To load the dataset splits for downstream tasks, you may follow the simple implementation below:

```{python}
from src.utils.helpers import load_data_splits, decode_dataset_texts

# get dataset splits
train, val, test, encoded = load_data_splits(path="data/small/small_data.pt") # change path as needed

# optionally, view short sections of the text with:
print("Training text preview:\n")
print(decode_dataset_text(train, max_chars=300))

print("\nValidation text preview:\n")
print(decode_dataset_text(val, max_chars=300))

print("\nTest text preview:\n")
print(decode_dataset_text(test, max_chars=300))
```

The next step after data preparation would be to set up the needed model hyper-parameters. We make this easy using YAML files. You may save your desired configuration anywhere as long as it follows the correct format provided, but for cleanliness we recommend saving all configurations to the `configs/` folder under `src/`. An example has been provided below for your reference:

```{yaml}
# configuration for SINGLE experiment
model:
  vocab_size: 27
  d_model: 128
  n_heads: 4
  d_ff: 512
  n_layers: 2
  max_len: 128
  drop: 0.3
train:
  experiment_name: "trial"
  epochs: 3
  lr: 0.00001
  batch_size: 32
  save_strategy: "steps"
  save_steps: 2500
```

Once done, you can move on to training the model on your prepared dataset. Note that here we only provide the basic parameters. Please refer to the source code <https://github.com/frznprograms/mini-transformer/blob/main/src/models/training.py> for more details and other ways to customise training:

```{python}
from src.models.training import ModelTrainer

mt = ModelTrainer(device="auto", config=config, seed=42)

mt.train(
    train_dataset=train,
    val_dataset=val,
    plot_loss=True,
)
```

You may also wish to run a grid search on different combinations of parameters. If so, you may specify a set of parameters in a YAML file as follows, adding and removing from the parameter lists as needed:

```{yaml}
grid_search_params:
  d_model: [64, 128]
  n_heads: [1, 2]
  n_layers: [1, 2]
  d_ff: [128, 256]
  lr: [0.00001, 0.00003]
  batch_size: [32, 64]

```

Then run a grid search on the specified parameters:

```{python}
from src.models.grid_search import GridSearchManager

config_file_path = "src/configs/my_configs.yaml"

search_manager = GridSearchManager(
    config_path=config_file_path, device="auto", seed=42
)

search_manager.run(train, val)
```

#### Project Structure

Please refer to the diagram below for an overview of our project structure and what each component handles.

```.
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
```

#### Conceptual Overview

#### References and Citations

In this section, we include references in our research, even if they were not used in the actual/final implementation.

```
@misc{hoffmann2022trainingcomputeoptimallargelanguage,
      title={Training Compute-Optimal Large Language Models},
      author={Jordan Hoffmann and Sebastian Borgeaud and Arthur Mensch and Elena Buchatskaya and Trevor Cai and Eliza Rutherford and Diego de Las Casas and Lisa Anne Hendricks and Johannes Welbl and Aidan Clark and Tom Hennigan and Eric Noland and Katie Millican and George van den Driessche and Bogdan Damoc and Aurelia Guy and Simon Osindero and Karen Simonyan and Erich Elsen and Jack W. Rae and Oriol Vinyals and Laurent Sifre},
      year={2022},
      eprint={2203.15556},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2203.15556},
}

@misc{alrfou2018characterlevellanguagemodelingdeeper,
      title={Character-Level Language Modeling with Deeper Self-Attention},
      author={Rami Al-Rfou and Dokook Choe and Noah Constant and Mandy Guo and Llion Jones},
      year={2018},
      eprint={1808.04444},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1808.04444},
}

@misc{vaswani2023attentionneed,
      title={Attention Is All You Need},
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762},
}

@misc{shaw2018selfattentionrelativepositionrepresentations,
      title={Self-Attention with Relative Position Representations},
      author={Peter Shaw and Jakob Uszkoreit and Ashish Vaswani},
      year={2018},
      eprint={1803.02155},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1803.02155},
}

@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864},
}

```
