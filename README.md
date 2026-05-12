# RL Exercise 1 Code Submission

This repository contains the practical Python code for the requested programming parts of Exercise 1.

## Setup

Use Python 3.9. The code was run locally with:

```powershell
py -3.9 -m pip install --user torch torchvision matplotlib gymnasium[classic-control] numpy
```

No special environment variables are required.

## Theory Question 3(e): Language Model Dynamic Programming

### Relevant Source/Input Files

- `language_model_dp.py`  
  Implements dynamic programming for finding the most probable word of a given length in the Bokoboko language. The transition probabilities from the assignment are encoded directly in the `TRANSITIONS` dictionary.

### Command

```powershell
py -3.9 .\language_model_dp.py
```

### Relevant Output Files

This command prints the answer to the terminal. It does not create an output file.

Expected result:

```text
Most probable word of size 5: BKBKO
Probability: 0.00676000
```

## Programming Question 1: MNIST

### Relevant Source/Input Files

- `mnist.py`  
  Part 1. The original skeleton-style file with the TODO training and evaluation sections completed. Uses logistic regression with SGD.

- `mnist_part2.py`  
  Part 2. Same linear model, but with an improved optimization configuration using Adam.

- `mnist_part3.py`  
  Part 3. A deeper model with one hidden layer of size 500 and a ReLU non-linearity.

- MNIST dataset  
  Downloaded automatically by `torchvision.datasets.MNIST` into the local `data/` directory when each script is run. The `data/` directory is ignored by Git.

### Commands

Each command is independent and can be run separately.

Part 1:

```powershell
py -3.9 .\mnist.py
```

Part 2:

```powershell
py -3.9 .\mnist_part2.py
```

Part 3:

```powershell
py -3.9 .\mnist_part3.py
```

### Relevant Output Files

- `outputs/mnist_loss_comparison.png`  
  Training-loss comparison plot for the three MNIST configurations.

- `outputs/mnist_summary.txt`  
  Summary of final test accuracies and final mean training losses.

The saved model files produced by the scripts, such as `model.pkl`, `model_part2.pkl`, and `model_part3.pkl`, are ignored by Git because they are generated artifacts.

Recorded results:

```text
Part 1: SGD baseline: final_accuracy=90 %, final_mean_loss=0.3802
Part 2: Adam config: final_accuracy=92 %, final_mean_loss=0.2226
Part 3: ReLU hidden=500: final_accuracy=98 %, final_mean_loss=0.0000
```

## Programming Question 2: CartPole Random Search

### Relevant Source/Input Files

- `cartpole_random_search.py`  
  Implements the CartPole agent, episode evaluation, one full random-search training run with 10000 sampled weight vectors, and the repeated 1000-search experiment requested for the histogram.

- `CartPole-v1` Gymnasium environment  
  Loaded through `gymnasium.make("CartPole-v1")`. No local input file is required.

### Command

```powershell
py -3.9 .\cartpole_random_search.py --num-searches 1000 --num-samples 10000 --target-score 200 --max-steps 200 --output-dir outputs
```

Parameter meanings:

- `--num-searches 1000`: repeat the random search 1000 times for the histogram experiment.
- `--num-samples 10000`: sample 10000 random weight vectors for the full training run, and use up to 10000 samples per repeated search.
- `--target-score 200`: score threshold for success.
- `--max-steps 200`: cap each episode at 200 steps, matching the assignment.
- `--output-dir outputs`: write result files under `outputs/`.

### Relevant Output Files

- `outputs/cartpole_random_search_histogram.png`  
  Histogram of the number of sampled episodes required until score 200 over 1000 searches.

- `outputs/cartpole_summary.txt`  
  Text summary of the CartPole random-search experiment.

Recorded results:

```text
Single full random-search training samples: 10000
Best score in full training: 200
First successful sample in full training: 7
Average episodes required until score 200: 13.51
Failed searches: 0
Best weights from full training: [ 0.2308 -0.2326  0.9944  0.9617]
```
