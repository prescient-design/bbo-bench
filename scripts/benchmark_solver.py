import json
import random

import boto3
import hydra
import numpy as np
import torch
import wandb
from bbo_bench.observers import SimpleObserver
from holo.logging import wandb_setup
from omegaconf import OmegaConf, open_dict
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from poli_baselines.solvers.simple.genetic_algorithm import (
    FixedLengthGeneticAlgorithm,
)


@hydra.main(
    version_base=None,
    config_path="../config/hydra",
    config_name="benchmark_optimizer",
)
def main(cfg):
    # Setup wandb and seeding
    wandb_setup(cfg)
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.set_float32_matmul_precision("medium")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Instantiate black box problem
    black_box = hydra.utils.instantiate(cfg.test_function, _convert_="partial")
    known_soln = black_box.optimal_solution()
    opt_val = black_box(known_soln)
    print("Black box info:", black_box.info)
    print("Black box optimal solution:", known_soln)

    # Generate initial solutions and optionally run presolver
    initial_solution = np.array(list(black_box.initial_solution()[0])).reshape(
        1, -1
    )
    random_seqs = np.array(
        [list(black_box.random_solution()[0]) for _ in range(127)]
    )
    x0 = np.concatenate([initial_solution, random_seqs], axis=0)
    y0 = black_box(x0)

    if cfg.run_presolver:
        presolver = FixedLengthGeneticAlgorithm(
            black_box=black_box,
            x0=x0,
            y0=y0,
            population_size=1000,
            prob_of_mutation=0.005,
        )

        presolver.solve(max_iter=2)
        presolver_x = np.array(presolver.history["x"])
        presolver_x = presolver_x.reshape(presolver_x.shape[0], -1)
        presolver_y = black_box(presolver_x)

    # If not running presolver, load solutions from data package
    else:
        bucket = cfg.ga_solution.bucket
        key_prefix = cfg.ga_solution.key_prefix
        s3_obj = boto3.client("s3")
        s3_clientobj_ehrlich = s3_obj.get_object(
            Bucket=bucket, Key=key_prefix + "ehrlich.jsonl"
        )
        s3_ehrlich_data = (
            s3_clientobj_ehrlich["Body"].read().decode("utf-8").splitlines()
        )
        ehrlich_data = [json.loads(line) for line in s3_ehrlich_data]

        s3_clientobj_presolved = s3_obj.get_object(
            Bucket=bucket, Key=key_prefix + "plain_pairs.jsonl"
        )
        s3_presolved_data = (
            s3_clientobj_presolved["Body"].read().decode("utf-8").splitlines()
        )
        presolved_data = [json.loads(line) for line in s3_presolved_data]

        # Sanity check that black box is the same between this experiment and the data package
        print("Data package black box info:", ehrlich_data)

        # Reformat solutions from data package to match black box API
        presolver_x = [record["particle"] for record in presolved_data]
        vocab = black_box.alphabet

        def convert_to_alphabet(x):
            new_x = []
            for elem in x:
                new_x.append([vocab[int(e)] for e in elem])
            return new_x

        presolver_x = [eval(x) for x in presolver_x]
        presolver_x = np.array(convert_to_alphabet(presolver_x))
        presolver_y = (
            -1
            * np.array(
                [(record["score"]) for record in presolved_data], dtype=float
            )[:, None]
        )

    # Instantiate observer to record metrics
    observer = SimpleObserver(cfg=cfg, opt_val=opt_val)
    black_box.set_observer(observer)

    observer.add_initial_sols(presolver_x, presolver_y)

    # Instantiate solver
    with open_dict(cfg.optimizer):
        if cfg.optimizer.name == "LaMBO2":
            logger = hydra.utils.instantiate(cfg.optimizer.trainer.logger)

            optimizer = LaMBO2(
                config=cfg.optimizer,
                black_box=black_box,
                x0=presolver_x,  # inconsistent API; fixed it
                y0=presolver_y.reshape(-1, 1),
                max_epochs_for_retraining=cfg.optimizer.max_epochs,
                logger=logger,
            )
        else:
            raise ValueError(
                "config optimizer name: {cfg.optimizer.name} is not currently supported"
            )

    # Run solver
    print(f"Searching for solution with optimal value {opt_val}...")
    optimizer.solve(max_iter=cfg.num_opt_steps)

    # Terminate black box and observer
    black_box.terminate()
    observer.finish(
        filename=cfg.output_dir
        + "/intermediate_sols/"
        + wandb.run.id
        + "_sols.json"
    )


if __name__ == "__main__":
    main()
