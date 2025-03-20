import random

import hydra
import numpy as np
import torch
from bbo_bench.attributions import cortex_model_to_value_function, shapley
from bbo_bench.observers import SimpleObserver
from bbo_bench.utils import add_vocab_to_lambo_cfg
from holo.logging import wandb_setup
from omegaconf import OmegaConf, open_dict
from poli_baselines.solvers.bayesian_optimization.lambo2 import LaMBO2
from poli_baselines.solvers.simple.genetic_algorithm import (
    FixedLengthGeneticAlgorithm,
)
from scipy.stats import spearmanr

import wandb


@hydra.main(
    version_base=None,
    config_path="../config/hydra",
    config_name="benchmark_shapley",
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
    black_box_copy = hydra.utils.instantiate(
        cfg.test_function, _convert_="partial"
    )
    known_soln = black_box.optimal_solution()
    opt_val = black_box(known_soln)
    print("Black box info:", black_box.info)
    print("Black box optimal solution:", known_soln)

    # Generate initial solutions and optionally run presolver
    initial_solution = np.array(list(black_box.initial_solution()[0])).reshape(
        1, -1
    )
    ref_seq = initial_solution.reshape(-1)
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
            population_size=200,
            prob_of_mutation=0.01,
        )

        presolver.solve(max_iter=1)
        presolver_x = np.array(presolver.history["x"])
        presolver_x = presolver_x.reshape(presolver_x.shape[0], -1)
        presolver_y = black_box(presolver_x)

    # If not running presolver, load solutions from data package
    else:
        assert cfg.presolved_data_package is not None
        presolver_x, presolver_y = hydra.utils.call(
            cfg.presolved_data_package,
            black_box=black_box,
        )

    # Instantiate observer to record metrics
    observer = SimpleObserver(cfg=cfg, opt_val=opt_val)
    black_box.set_observer(observer)

    observer.add_initial_sols(presolver_x, presolver_y)

    # Instantiate solver
    with open_dict(cfg.optimizer):
        if cfg.optimizer.name == "LaMBO2":
            # Instantiate a logger
            logger = hydra.utils.instantiate(cfg.optimizer.trainer.logger)

            # Make LaMBO2 config use the same vocab as the black box
            vocab = black_box.alphabet
            cfg = add_vocab_to_lambo_cfg(cfg, vocab)

            # Instantiate LaMBO2 optimizer
            optimizer = LaMBO2(
                config=cfg.optimizer,
                black_box=black_box,
                x0=presolver_x,
                y0=presolver_y.reshape(-1, 1),
                max_epochs_for_retraining=cfg.optimizer.max_epochs,
                logger=logger,
            )
        else:
            raise ValueError(
                f"config optimizer name: {cfg.optimizer.name} is not currently supported"
            )

    # Run solver and compare shapley values
    # define reference sequence
    # ref_seq = np.array(optimizer.get_candidate_points()[0].split(" "))
    print(f"Searching for solution with optimal value {opt_val}...")
    # optimizer.solve(max_iter=cfg.num_opt_steps)
    for i in range(
        cfg.num_opt_steps
    ):  # changed this from optimizer.solve(max_iter=cfg.num_opt_steps)
        new_designs, new_y = optimizer.step()

        # Analyze shapley values
        # new design with highest new_y value
        example_seq = np.array(optimizer.get_candidate_points()[0].split(" "))
        new_design_example_seq = np.array(list(new_designs[0]))

        print("Reference sequence:", ref_seq)
        print("Example sequence:", example_seq)
        print("New design example sequence:", new_design_example_seq)

        # ref_seq = initial_solution[0]
        # model = hydra.utils.instantiate(optimizer.cfg.tree)
        # model.build_tree(optimizer.cfg, skip_task_setup=True)
        # model.load_state_dict(
        #    torch.load(
        #        optimizer.model_path,
        #        map_location="cpu",
        #        weights_only=False,
        #    )["state_dict"]
        # )
        model = optimizer.model
        model_value_func = cortex_model_to_value_function(model)
        model_feasibility_func = cortex_model_to_value_function(
            model, task="generic_constraint"
        )

        def black_box_feasibility_func(seqs):
            return np.where(black_box_copy(seqs) > float("-inf"), 1, 0)

        shapley_model = shapley(
            model_value_func,
            example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=black_box_feasibility_func,
        )
        shapley_black_box = shapley(
            black_box_copy,
            example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=black_box_feasibility_func,
        )
        shapley_model_feasibility = shapley(
            model_feasibility_func,
            example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=None,
        )
        shapley_black_box_feasibility = shapley(
            black_box_feasibility_func,
            example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=None,
        )

        alternative_shapley_model = shapley(
            model_value_func,
            new_design_example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=black_box_feasibility_func,
        )
        alternative_shapley_black_box = shapley(
            black_box_copy,
            new_design_example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=black_box_feasibility_func,
        )
        alternative_shapley_model_feasibility = shapley(
            model_feasibility_func,
            new_design_example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=None,
        )
        alternative_shapley_black_box_feasibility = shapley(
            black_box_feasibility_func,
            new_design_example_seq,
            ref_seq,
            method="exact",
            feasibility_mask_function=None,
        )

        def get_shapley_metrics(
            shapley_model, shapley_black_box, idx, prefix=""
        ):
            shapley_spearman = spearmanr(shapley_model, shapley_black_box)[0]
            shapley_mse = np.mean((shapley_model - shapley_black_box) ** 2)
            shapley_mae = np.mean(np.abs(shapley_model - shapley_black_box))
            shapley_max_error = np.max(
                np.abs(shapley_model - shapley_black_box)
            )
            shapley_vstack = np.vstack((shapley_black_box, shapley_model)).T
            shapley_table = wandb.Table(
                data=shapley_vstack,
                columns=["shapley_black_box", "shapley_model"],
            )
            shapley_table.add_column("index", list(range(len(shapley_model))))
            metrics = {
                f"{prefix}/shapley_spearman": shapley_spearman,
                f"{prefix}/shapley_mse": shapley_mse,
                f"{prefix}/shapley_mae": shapley_mae,
                f"{prefix}/shapley_max_error": shapley_max_error,
                f"{prefix}/shapley_table": shapley_table,
                f"{prefix}/shapley_black_box_sum": np.sum(shapley_black_box),
                f"{prefix}/shapley_model_sum": np.sum(shapley_model),
                f"{prefix}/outer_loop_step": idx,
            }
            return metrics

        wandb.log(
            get_shapley_metrics(
                shapley_model, shapley_black_box, i, "attributions_value"
            )
        )
        wandb.log(
            get_shapley_metrics(
                shapley_model_feasibility,
                shapley_black_box_feasibility,
                i,
                "attributions_feasibility",
            )
        )
        wandb.log(
            get_shapley_metrics(
                alternative_shapley_model,
                alternative_shapley_black_box,
                i,
                "attributions_value_alt",
            )
        )
        wandb.log(
            get_shapley_metrics(
                alternative_shapley_model_feasibility,
                alternative_shapley_black_box_feasibility,
                i,
                "attributions_feasibility_alt",
            )
        )

        print("Shapley values for model:", shapley_model)
        print("Shapley values for black box:", shapley_black_box)

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
