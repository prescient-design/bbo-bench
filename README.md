# bbo-bench
Scripts and tools to quickly run black box optimization experiments using the [poli](https://machinelearninglifescience.github.io/poli-docs/index.html) ecosystem.

## Project Overview
bbo-bench serves as a lightweight experimentation wrapper around the [poli-core](https://github.com/MachineLearningLifeScience/poli.git) and [poli-baselines](https://github.com/MachineLearningLifeScience/poli-baselines.git) repositories. Rather than implementing optimizers or test functions directly, this repo provides:

1. Ready-to-use scripts for running benchmarking experiments
2. Hydra-based configuration management for experiment settings
3. Wandb integration for experiment tracking and visualization
4. Utilities for loading and processing benchmarking data

The main value of this repo is in streamlining the experimental workflow, allowing researchers to quickly set up and run benchmarking experiments without boilerplate code.

## Installation

From the root of the repo, do:
```bash
pip install -e .
```
## Usage

To run the benchmarking script with the default settings specified in the configs, run:
```bash
python scripts/benchmark_optimizer.py
```
This will log results in a [wandb](https://wandb.ai/site) run. You may need to log in to your wandb account following a command line prompt and/or update the config found at `config/hydra/benchmark_optimizer.yaml` with the correct wandb host address.

### Config files

The central config used by the benchmarking script is `config/hydra/benchmark_optimizer.yaml`. Note that we use `hydra` to compose hierarchical configs, so this central config specifies a default list of nested configs for all the important components of a benchmarking run:
- `optimizer`
    - this is the black box optimization algorithm to benchmark
    - by default, LaMBO-2 with parameters specified in `config/hydra/optimizer/lambo2.yaml`
    - the above LaMBO-2 config uses further nested configs found in subfolders of `config/hydra/optimizer`
- `test_function`
    - this is the black box test function used to evaluate the optimizers
    - by default, an Ehrlich function with sequence length 32, vocab size 32, and 4 motifs of length 4
    - alternative example Ehrlich functions can be found in `config/hydra/test_function`, and others can easily be defined following this pattern. See [Stanton et al. 2024](https://arxiv.org/abs/2407.00236) for guidance on Ehrlich function parameters to scale the difficulty of the problem.
- `presolved_data_package`
    - this is an optional component to provide initial training data to your optimizer
    - by default, we include a url to downloadable data to replicate our experiments benchmarking LaMBO-2
    - if you prefer not to provide training data yourself, you can instead let the benchmarking script create its own training data by setting `run_presolver` to `True`, which will use a simple genetic algorithm to generate initial solutions to pass to the optimizer you are benchmarking

You can change these configs and add new ones to run your desired benchmarking experiment.

To run a wandb sweep across different parameter values, see `config/wandb/example_sweep.yaml` for an example. Run:
```bash
wandb sweep config/wandb/example_sweep.yaml
```
to initialize this sweep, then run the command that wandb outputs (i.e. `wandb agent <sweep>`).

### Observers
The benchmarking script provided in this repo isolates much of the logging functionality in observers, which track metrics every time the black box function is queried.
The `SimpleObserver` class found in `src/bbo_bench/observers/_simple_observer.py` is the default observer, and it tracks metrics such as simple regret, cumulative regret, and diversity of proposed solutions. You can add other metrics by adding methods to `SimpleObserver` or defining a new observer class.

Observers are attached to the black box, so they only track metrics for solutions submitted to the black box for evaluation. Other metrics (for example, model training logs for a model within your optimizer) can be logged independently. As an example, the LaMBO-2 optimizer by default tracks internal metrics with a [lightning WandbLogger](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.loggers.wandb.html).

## Running experiments with a different optimizer
To run experiments with a different optimizer from the [poli-baselines](https://github.com/MachineLearningLifeScience/poli-baselines.git) repo:

1. Create a new config file in config/hydra/optimizer/ for your chosen optimizer
2. Update the main config to use your new optimizer:
```bash
python scripts/benchmark_optimizer.py optimizer=your_optimizer
```

Note that currently the benchmarking script is set up specifically for LaMBO-2. To use other optimizers from poli-baselines, you'll need to modify the `benchmark_optimizer.py` script to instantiate your optimizer of choice instead of LaMBO-2.

If you've developed a new optimizer that's not in poli-baselines, follow these guidelines to integrate it with the benchmarking suite::
- Input format
    - The optimizer should accept input data (from either the data package or the genetic algorithm presolver) as NumPy arrays.
    - An example of Ehrlich function sequences and evaluation scores:
        ```python
        inputs = numpy.array([['M', 'A', 'A', 'S', 'T', 'Q'],
                              ['M', 'A', 'A', 'S', 'S', 'S']])
        scores = numpy.array([[0.0],
                              [0.2]])
        ```
- Output format
    - The optimizer should output solutions, i.e. query the black box, with NumPy arrays with the same formatting as above

- Match API with `AbstractSolver` from `poli-baselines`
    - The [AbstractSolver](https://github.com/MachineLearningLifeScience/poli-baselines/blob/main/src/poli_baselines/core/abstract_solver.py) class has a very lightweight API; for consistency with the other optimizers benchmarked, it is advisable to implement a `solve` method in your optimizer to match this API.

Then, within the `benchmark_optimizer.py` script, in the section that currently instantiates the LaMBO-2 optimizer, you can instead instantiate the new optimizer.

## Contributing

Contributions are welcome!

### Install dev requirements and pre-commit hooks

```bash
pip install -r requirements-dev.in
pre-commit install
```
