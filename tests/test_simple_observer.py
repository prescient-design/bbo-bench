import numpy as np
from bbo_bench.observers import SimpleObserver


class TestSimpleObserver:
    def setup_method(self):
        # Create a mock config
        class MockConfig:
            def __init__(self):
                self.optimizer = MockOptimizerConfig()
                self.log_interval = 1
                self.save_intermediate_sols = False

        class MockOptimizerConfig:
            def __init__(self):
                self.num_samples = 2

        self.cfg = MockConfig()
        self.opt_val = 1.0
        self.observer = SimpleObserver(cfg=self.cfg, opt_val=self.opt_val)

    def test_initialization(self):
        assert self.observer.x_s == []
        assert self.observer.y_s == []
        assert self.observer.rounds == []
        assert self.observer.step == 0
        assert self.observer.cumulative_regret == 0.0
        assert self.observer.best_reward == 0

    def test_add_initial_sols(self):
        # Create mock data
        xs = np.array([["A", "B", "C"], ["D", "E", "F"]])
        ys = np.array([[0.5], [0.7]])

        # Add initial solutions
        self.observer.add_initial_sols(xs, ys)

        # Check that data was added correctly
        assert len(self.observer.x_s) == 2
        assert len(self.observer.y_s) == 2
        assert len(self.observer.rounds) == 2
        assert all(round == -1 for round in self.observer.rounds)

    def test_frac_unique(self):
        # Create mock data
        sol1 = np.array(["ABCD"]).reshape(1, 1)
        sol2 = np.array(["ABCD"]).reshape(1, 1)
        sol3 = np.array(["EFGH"]).reshape(1, 1)

        # Test all unique
        assert self.observer.get_frac_unique([sol1, sol3]) == 1.0

        # Test some duplicates
        assert self.observer.get_frac_unique([sol1, sol2, sol3]) == 2 / 3

    def test_diversity(self):
        # Create mock data with known Levenshtein distances
        sol1 = np.array(["ABCD"]).reshape(1, 1)
        sol2 = np.array(["ABCE"]).reshape(1, 1)  # Distance 1 from sol1
        sol3 = np.array(["WXYZ"]).reshape(1, 1)  # Distance 4 from others

        # Test diversity calculation
        diversity = self.observer.get_diversity([sol1, sol2, sol3])

        # Expected: (1 + 4 + 4) / 3 = 3
        assert diversity == 3.0

    def test_observe(self):
        # Setup
        x = np.array(["ABCD"]).reshape(1, 1)
        y = np.array([0.8]).reshape(1, 1)

        # Initial state
        assert self.observer.step == 0
        assert self.observer.best_reward == 0

        # Observe a new solution
        self.observer.observe(x, y)

        # Check updates
        assert self.observer.step == 1
        assert len(self.observer.x_s) == 1
        assert len(self.observer.y_s) == 1
        assert self.observer.best_reward == 0.8
        assert self.observer.cumulative_regret == self.opt_val - 0.8
