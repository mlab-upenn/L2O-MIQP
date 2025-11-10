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

### Training the Model

To train a model, navigate to the corresponding example directory and run:
```
python test.py
```

You can pass command-line arguments to customize the training.
Use `python test.py --help` to display the available options.
An example usage can be found in `run.sh`.
The trained model will be saved in the same directory with the filename `model_{your_input_filename}.pth` and the validation results are saved in `stats_{your_input_filename}.pt`.

For example:
```
# supervised learning (default)
python test.py --save_stats --filename robot_nav_sl
# self-supervised learning
python test.py --w_obj 1e-5 --w_slack 1.0 --w_con 1.0 --w_sup 0.0 --save_stats --filename robot_nav_ssl 
# hybrid learning
python test.py --w_obj 0.0 --w_slack 0.0 --w_con 1.0 --w_sup 1e3 --save_stats --filename robot_nav_hl_1
```

then the saved model is `model_{filename}.pth`, and the validation results are saved in `stats_{filename}.pt`.

You can train multiple models (e.g., supervised learning, self-supervised learning, or hybrid L2O with different weight configurations), by modifying the `run.sh` script as needed, then execute:
```
bash run.sh
```

### Evaluating the trained model

After training, you can evaluate the saved model again by running:
```
python evaluate.py --filename $your_saved_model
```

Replace $your_saved_model with the model name used during training — excluding the prefix `model_` and the file extension `.pth`.

For example, if your trained file is named `model_robot_nav_sl.pth`, run:
```
python evaluate.py --filename robot_nav_sl

```