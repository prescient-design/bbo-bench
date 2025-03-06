# bbo-bench
Scripts and tools to benchmark black box optimization algorithms.

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
This will log results in a wandb run. You may need to log in to your wandb account following a command line prompt and/or update the config found at `config/hydra/benchmark_optimizer.yaml` with the correct wandb host address.

### Understanding configs

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

If you want to run a wandb sweep across different values of certain config parameters, this is straightforward -- see `config/wandb/example_sweep.yaml` for an example of how to design a sweep over a few LaMBO-2 parameters. You can run `wandb sweep config/wandb/example_sweep.yaml` to initialize this sweep, then run the command that wandb outputs (i.e. `wandb agent <sweep>`).

### Understanding observers


## Benchmarking a new optimizer

To integrate a new black box optimizer into the benchmarking suite, follow the guidelines below to ensure compatibility:
- Input format
    - The optimizer should accept input data (from either the data package or the genetic algorithm presolver) as NumPy arrays.
    - An example Ehrlich function datapoint is: `np.array(['MAASTQAV'])`

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
