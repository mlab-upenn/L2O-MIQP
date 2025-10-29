# SS_Learning_MIQP

This repository contains the code for the project *Learning to Optimize (L2O)* applied to mixed-integer quadratic programming (MIQP).

## Overview

We implemented two models that leverage a quadratic programming (QP) layer to:
1. **Embed** the dependence between integer and continuous decision variables during training.
2. **Correct** the integer solution of a supervised learning neural network.

## Installation

To install the package dependencies, run:
```bash
pip install -r requirements.txt
```

To install the package, go to the root directory of the repository and execute
```bash
pip install -e . 
```

## Example 

In `examples`, we provide two notebooks:
- [test_incorporated_qplayer.ipynb](examples/test_incorporated_qplayer.ipynb)
- [test_corrected_qplayer.ipynb](examples/test_corrected_qplayer.ipynb)

These demonstrate the use of our frameworks for a simple MIQP problem.
 