import json
from pathlib import Path
from typing import Iterable

import Levenshtein
import numpy as np
import wandb
from poli.core.black_box_information import BlackBoxInformation
from poli.core.util.abstract_observer import AbstractObserver


class SimpleObserver(AbstractObserver):
    def __init__(self, cfg, opt_val) -> None:
        self.x_s = []
        self.y_s = []
        self.rounds = []
        self.cfg = cfg
        self.opt_val = opt_val
        self.step = 0
        self.cumulative_regret = 0.0
        self.best_reward = 0
        self.running_rewards = []
        self.running_sols = []
        super().__init__()
        self.initial_sols = None
        self.outer_loop_step = 0

    def initialize_observer(
        self,
        problem_setup_info: BlackBoxInformation,
        caller_info: object,
        seed: int,
    ) -> object:
        ...

    def get_frac_unique(self, sol_list: Iterable) -> float:
        """
        Compute the fraction of sequences that are unique in a list

        Args:
            sol_list: Iterable of sequences

        Returns:
            Number of unique sequences divided by number of total sequences
        """
        sol_list_strings = [x.flatten().item() for x in sol_list]
        return len(set(sol_list_strings)) / len(sol_list)

    def get_diversity(self, sol_list: Iterable):
        """
        Compute the average Levenshtein distance between every pair of sequences in a list

        Args:
            sol_list: Iterable of sequences

        Returns:
            Average Levenshtein distance between every pair of sequences
        """
        sol_list_strings = [x.flatten().item() for x in sol_list]
        distances = []
        for i in range(len(sol_list_strings)):
            for j in range(i + 1, len(sol_list_strings)):
                distances.append(
                    Levenshtein.distance(
                        sol_list_strings[i], sol_list_strings[j]
                    )
                )
        return (
            2
            * np.sum(distances)
            / (len(sol_list_strings) * (len(sol_list_strings) - 1))
        )

    def exp_lev_kernel(self, seq1, seq2, gamma=1.0):
        """
        Compute the exponential Levenshtein kernel between two sequences.

        Args:
            seq1: First sequence
            seq2: Second sequence
            gamma: Parameter for the kernel (default is 1.0)

        Returns:
            Kernel value between seq1 and seq2
        """
        distance = Levenshtein.distance(seq1, seq2)
        return np.exp(-gamma * distance)

    def compute_mmd(self, X, Y, gamma=1.0 / 10):
        """
        Compute Maximum Mean Discrepancy between samples X and Y.

        Args:
            X: list of samples from first distribution
            Y: list of samples from second distribution
            gamma: parameter for RBF kernel (1/(2*σ²))

        Returns:
            Estimated MMD value
        """
        if X is None or Y is None:
            return -1, -1, -1
        X = [x.flatten().item() for x in X]
        Y = [y.flatten().item() for y in Y]
        m = len(X)
        n = len(Y)

        # sum of K_XX entries
        xx_sum = 0
        for i in range(m):
            for j in range(i + 1, m):
                xx_sum += self.exp_lev_kernel(X[i], X[j], gamma)
        mmd_squared = xx_sum / (m * (m - 1) / 2)

        # sum of K_YY entries
        yy_sum = 0
        for i in range(n):
            for j in range(i + 1, n):
                yy_sum += self.exp_lev_kernel(Y[i], Y[j], gamma)
        mmd_squared += yy_sum / (n * (n - 1) / 2)

        # sum of K_XY entries
        xy_sum = 0
        for i in range(m):
            for j in range(n):
                xy_sum += self.exp_lev_kernel(X[i], Y[j], gamma)
        mmd_squared -= 2 * xy_sum / (m * n)

        return (
            xx_sum / (m * (m - 1) / 2),
            yy_sum / (n * (n - 1) / 2),
            xy_sum / (m * n),
        )  # Return MMD components

    def add_initial_sols(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """
        Add initial solutions to the observer

        Args:
            xs: Initial solutions
            ys: Initial rewards
        """
        assert len(xs) == len(ys)
        for x in xs:
            self.x_s.append(np.array("".join(x))[None, None])
            self.rounds.append(-1)
        for y in ys:
            self.y_s.append(y[None, :])

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        """
        Observe a new solution and its reward and log the data with wandb

        Args:
            x: New solution submitted to black box as a query
            y: Reward of the new solution
            context: Context of the solution
        """
        self.step += 1
        self.x_s.append(x)
        self.y_s.append(y)
        self.rounds.append(self.outer_loop_step)

        # update best rewards
        last_reward = max(y)
        if last_reward > self.best_reward:
            self.best_reward = last_reward
        simple_regret_best = self.opt_val - self.best_reward
        self.cumulative_regret += self.opt_val - self.best_reward

        # update list of running rewards and solutions to compute running metrics
        if len(self.running_rewards) >= self.cfg.optimizer.num_samples:
            self.running_rewards.pop(0)
        self.running_rewards.append(last_reward)
        if len(self.running_sols) >= self.cfg.optimizer.num_samples:
            self.running_sols.pop(0)
        self.running_sols.append(x)

        if (
            self.step
            % (self.cfg.log_interval * self.cfg.optimizer.num_samples)
            == 0
        ):
            if self.initial_sols is None:
                self.initial_sols = self.running_sols.copy()
            xx_mean, yy_mean, xy_mean = self.compute_mmd(
                self.running_sols, self.initial_sols
            )
            self.outer_loop_step += 1

            metrics = {
                "black_box/simple_regret_best": simple_regret_best,
                "black_box/cumulative_regret": self.cumulative_regret,
                "black_box/running_average_regret": 1
                - float(np.ma.masked_invalid(self.running_rewards).mean()),
                "black_box/running_fraction_feasible": float(
                    np.mean(np.array(self.running_rewards) > float("-inf"))
                ),
                "black_box/running_fraction_unique": self.get_frac_unique(
                    self.running_sols
                ),
                "black_box/running_diversity": self.get_diversity(
                    self.running_sols
                ),
                "black_box/mmd_to_initial_sols": np.sqrt(
                    max(xx_mean + yy_mean - 2 * xy_mean, 0)
                ),
                "black_box/timestep": self.step,
                "black_box/outer_loop_step": self.outer_loop_step,
            }
            wandb.log(metrics)

    def finish(self, filename):
        """
        Finish and save the observer data to a file

        Args:
            filename: Name of the file to save the data
        """
        if self.cfg.save_intermediate_sols:
            x_s = np.concat(self.x_s, axis=0)[:, 0]
            y_s = np.concat(self.y_s, axis=0)[:, 0]
            x_s = [str(x) for x in x_s]
            y_s = [float(y) for y in y_s]
            print(x_s)
            print(y_s)
            output_dir = Path(filename).parent
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            with open(filename, "w") as f:
                json.dump({"x_s": x_s, "y_s": y_s, "rounds": self.rounds}, f)
            print("Finished and saved observer data")
        else:
            print("Finished")
