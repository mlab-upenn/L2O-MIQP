# L2O-MIQP

This repository contains the code for the project *Learning to Optimize (L2O)* applied to mixed-integer quadratic programming (MIQP).

## Overview

We implemented a L2O framework that leverages a quadratic programming (QP) layer to embed the dependence between integer and continuous decision variables, and utilizes a hybrid loss function combining supervised and self-supervised losses to leverage the strengths of both during training.

## Installation

To install the package dependencies, run:
```bash
pip install -r requirements.txt
```

To install the package, go to the root directory of the repository and execute
```bash
pip install -e . 
```

## Examples

We provide two example problems:
1. [**Collision avoidance for robot navigation**](examples/robot_nav)
2. [**Simplified thermal energy tank system**](examples/energy)

Each example includes scripts for training and evaluating different learning models.

### Datasets

The datasets for both examples are included in the repository under the `data/` directory.

To generate new datasets, run the following scripts:
- For robot navigation:
  ```
  python generate_data.py --num_samples <number_of_simulations>
  ```
  Additional flags include `--autosave`, `--shuffle`, `--horizon`, and `--save_path` for configuring the simulation horizon and storage behavior.
- For energy system:
  ```
  python generate_data.py --num_samples <number_of_simulations>
  ```
  Additional flags include `--horizon`, `--seed`, and `--save_path` for finer control over the generated dataset.


### Training the Model

To train a model, navigate to the corresponding example directory and run:
```
python train.py
```

You can pass command-line arguments to customize the training.
Use `python train.py --help` to display the available options.
An example usage can be found in `run.sh`.
The trained model will be saved in the same directory with the filename `model_{your_input_filename}.pth` and the validation results are saved in `stats_{your_input_filename}.pt`.

For example:
```
# supervised learning (default)
python train.py --save_stats --filename robot_nav_sl
# self-supervised learning
python train.py --w_obj 1e-5 --w_slack 1.0 --w_con 1.0 --w_sup 0.0 --save_stats --filename robot_nav_ssl
# hybrid learning
python train.py --w_obj 0.0 --w_slack 0.0 --w_con 1.0 --w_sup 1e3 --save_stats --filename robot_nav_hl_1
```

then the saved model is `model_{filename}.pth`, and the validation results are saved in `stats_{filename}.pt`.

You can train multiple models (e.g., supervised learning, self-supervised learning, or hybrid L2O with different weight configurations), by modifying the `run.sh` script as needed, then execute:
```
bash run.sh
```

### Evaluating the trained model

After training, you can evaluate checkpoints (from inside each example directory) with:
```
python evaluate.py --model checkpoints/model_$run_name.pth
```
Replace `$run_name` with the name you supplied during training. You can also pass just the base name (e.g., `--model robot_nav_sl`), and the script will look in `checkpoints/` automatically while writing stats to `checkpoints/stats_<model>.pt`. Use `--stats_out custom/path.pt` if you want to override the output location.
