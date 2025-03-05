# bbo-bench
Scripts and tools to benchmark black box optimization algorithms.

## Installation

From the root of the repo, do:
```bash
pip install -e .
```
## Run the benchmark script

To run the benchmarking script with the default settings specified in the configs, run:
```bash
python scripts/benchmark_solver.py
```
This will log results in a wandb run. You may need to log in to your wandb account following a command line prompt and/or update the config found at `config/hydra/benchmark_optimizer.yaml` with the correct wandb host address.

### Understanding configs
The central config used by the benchmarking script is `config/hydra/benchmark_optimizer.yaml`. Note that we use `hydra` to compose hierarchical configs, so this central config specifies nested configs for the central components of a benchmarking run: 
- `optimizer` (by default LaMBO-2 with parameters specified in `config/hydra/optimizer/lambo2.yaml`)
- `test_function` (by default an Ehrlich function, alternative example Ehrlich functions can be found in `config/hydra/test_function`)
- `presolved_data_package` (by default a url to downloadable data)

You can change these configs to run your desired benchmarking experiment. You may also want to run a sweep across different values of certain config parameters. The script works well with wandb sweeps; for example, `config/wandb/example_sweep.yaml` specifies a sweep over a few LaMBO-2 parameters. You can run `wandb sweep config/wandb/example_sweep.yaml` to initialize this sweep, then run the command that wandb outputs (i.e. `wandb agent <sweep>`). 

## Benchmarking a new model
 
