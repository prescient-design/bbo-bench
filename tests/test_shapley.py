"""
Tests for efficient Shapley value implementation using pytest.

Run with: pytest -xvs test_efficient_shapley.py
"""

import time

import numpy as np
import pytest

# Add parent directory to path to import the efficient_shapley module
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import the shapley functions from the module
from bbo_bench.attributions._shapley import (
    cortex_model_to_value_function,
    exact_shapley,
    kernel_shapley,
    mcmc_shapley,
    optimize_batch_size,
    permutation_sampling_shapley,
    shapley,
)


# Fixtures for test data
@pytest.fixture
def simple_test_data():
    """Fixture for simple test data."""

    def value_fn(x):
        return np.array([sum(s == "B" for s in seq) for seq in x])

    final = np.array(["B", "B", "A"])
    ref = np.array(["A", "A", "A"])
    expected = np.array([1.0, 1.0, 0.0])
    return value_fn, final, ref, expected


@pytest.fixture
def xor_test_data():
    """Fixture for XOR function test data."""

    def xor_value_fn(x):
        """Value function with feature interactions - rewards if an odd number of features is 'B'."""
        return np.array(
            [1.0 if sum(s == "B" for s in seq) % 2 == 1 else 0.0 for seq in x]
        )

    final = np.array(["B", "B", "B"])
    ref = np.array(["A", "A", "A"])
    expected = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
    return xor_value_fn, final, ref, expected


@pytest.fixture
def complex_test_data():
    """Fixture for complex test data with multiple features and interactions."""

    def inner_fn(vector):  # B x N
        if vector[0] == "B":
            if vector[1] == "C":
                return 1
            else:
                return 0
        elif vector[2] == "D":
            if vector[1] == "A":
                return 1
            else:
                return 0.6
        else:
            return 0

    def complex_value_fn(batches):
        return np.array([inner_fn(batch) for batch in batches])

    final = np.array(["B", "C", "D"])
    ref = np.array(["A", "A", "A"])
    expected = np.array([0.13333333, 0.433333333, 0.43333333])
    return complex_value_fn, final, ref, expected


@pytest.fixture
def long_test_data():
    """Fixture for longer sequence test data."""
    np.random.seed(42)  # For reproducibility

    def random_value_fn(x):
        """Random value function based on presence of specific tokens at specific positions."""
        values = np.zeros(len(x))
        for i, seq in enumerate(x):
            # Sum contributions from each position
            for pos, token in enumerate(seq):
                if (
                    token == "B" and pos % 4 == 0
                ):  # 'B' at positions 0, 4, 8, ...
                    values[i] += 0.5
                elif (
                    token == "C" and pos % 4 == 1
                ):  # 'C' at positions 1, 5, 9, ...
                    values[i] += 0.3
                elif (
                    token == "D" and pos % 4 == 2
                ):  # 'D' at positions 2, 6, 10, ...
                    values[i] += 0.2
                elif (
                    token == "E" and pos % 4 == 3
                ):  # 'E' at positions 3, 7, 11, ...
                    values[i] += 0.1
                # Add some interactions
                if pos > 0 and seq[pos] == "B" and seq[pos - 1] == "C":
                    values[i] += 0.25
        return values

    final = np.array(["B", "C", "D", "E"] * 4)  # 16 elements
    ref = np.array(["A"] * 16)
    return random_value_fn, final, ref


@pytest.fixture
def mock_cortex_model():
    """Fixture for a mock Cortex model."""

    class MockCortexModel:
        def __init__(self):
            ...
            # Mock implementation

        def call_from_str_array(self, str_array, corrupt_frac):
            # Simulate model prediction
            class TreeOutput:
                def __init__(self, values):
                    self.values = values

                def fetch_task_outputs(self, task_key):
                    # Return a dict with "loc" key
                    class Tensor:
                        def __init__(self, values):
                            self.values = values

                        def squeeze(self, dim):
                            return self

                        def mean(self, dim):
                            # Mean across ensemble dimension
                            return Tensor(np.mean(self.values, axis=0))

                        def detach(self):
                            return self

                        def cpu(self):
                            return self

                        def numpy(self):
                            return self.values

                    # Calculate some value based on the sequence
                    values = []
                    for seq in str_array:
                        # Count occurrences of 'B' and 'C'
                        score = seq.count("B") * 0.5 + seq.count("C") * 0.3
                        values.append(score)

                    # Simulate ensemble of 2 models
                    ensemble_values = np.array([values, values])
                    return {"loc": Tensor(ensemble_values)}

            return TreeOutput(str_array)

    return MockCortexModel()


# Tests for exact shapley (if available)
def test_exact_shapley_simple(simple_test_data):
    """Test that the exact Shapley function works for a simple case."""
    value_fn, final, ref, expected = simple_test_data
    # Compute exact Shapley values for the simple case
    exact_values = exact_shapley(value_fn, final, ref)

    # Check results
    np.testing.assert_allclose(exact_values, expected, atol=1e-6)


def test_exact_shapley_xor(xor_test_data):
    """Test that the exact Shapley function works for an XOR case."""

    value_fn, final, ref, expected = xor_test_data
    # Compute exact Shapley values
    exact_values = exact_shapley(value_fn, final, ref)

    # Check results
    np.testing.assert_allclose(exact_values, expected, atol=1e-6)


def test_exact_shapley_complex(complex_test_data):
    """Test that the exact Shapley function works for a complex case."""
    value_fn, final, ref, expected = complex_test_data
    # Compute exact Shapley values
    exact_values = exact_shapley(value_fn, final, ref)

    # Check results
    np.testing.assert_allclose(exact_values, expected, atol=1e-6)
    # Check that sum of attributions equals total value difference


# Tests for mcmc_shapley
def test_mcmc_shapley_simple(simple_test_data):
    """Test that the mcmc_shapley function works for a simple case."""
    value_fn, final, ref, expected = simple_test_data

    # Use a high number of samples for accurate results
    values = mcmc_shapley(value_fn, final, ref, n_samples=1000, seed=42)

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.02)


def test_mcmc_shapley_xor(xor_test_data):
    """Test that the mcmc_shapley function works for an XOR case."""
    value_fn, final, ref, expected = xor_test_data

    # Use a high number of samples for accurate results
    values = mcmc_shapley(value_fn, final, ref, n_samples=1000, seed=42)

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.02)


def test_mcmc_shapley_complex(complex_test_data):
    """Test that the mcmc_shapley function works for a complex case."""
    value_fn, final, ref, expected = complex_test_data

    # Use a high number of samples for accurate results
    values = mcmc_shapley(value_fn, final, ref, n_samples=5000, seed=42)

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.01)
    # Check that sum of attributions equals total value difference
    total_diff = (
        value_fn(final.reshape(1, -1))[0] - value_fn(ref.reshape(1, -1))[0]
    )
    assert np.isclose(values.sum(), total_diff)


# Tests for permutation_sampling_shapley
def test_permutation_sampling_shapley_simple(simple_test_data):
    """Test that the permutation_sampling_shapley function works for a simple case."""
    value_fn, final, ref, expected = simple_test_data

    values = permutation_sampling_shapley(
        value_fn, final, ref, n_samples=500, seed=42
    )

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.02)


def test_permutation_sampling_shapley_xor(xor_test_data):
    """Test that the permutation_sampling_shapley function works for an XOR case."""
    value_fn, final, ref, expected = xor_test_data

    values = permutation_sampling_shapley(
        value_fn, final, ref, n_samples=500, seed=42
    )

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.02)


def test_permutation_sampling_shapley_complex(complex_test_data):
    """Test that the permutation_sampling_shapley function works for a complex case."""
    value_fn, final, ref, expected = complex_test_data

    values = permutation_sampling_shapley(
        value_fn, final, ref, n_samples=5000, seed=42
    )

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.01)
    # Check that sum of attributions equals total value difference
    total_diff = (
        value_fn(final.reshape(1, -1))[0] - value_fn(ref.reshape(1, -1))[0]
    )
    assert np.isclose(values.sum(), total_diff)


# Tests for kernel_shapley
def test_kernel_shapley_simple(simple_test_data):
    """Test that the kernel_shapley function works for a simple case."""
    value_fn, final, ref, expected = simple_test_data

    values = kernel_shapley(value_fn, final, ref, n_samples=500, seed=42)

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.1)


def test_kernel_shapley_xor(xor_test_data):
    """Test that the kernel_shapley function works for an XOR case."""
    value_fn, final, ref, expected = xor_test_data

    values = kernel_shapley(value_fn, final, ref, n_samples=10000, seed=42)

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.15)


def test_kernel_shapley_complex(complex_test_data):
    """Test that the kernel_shapley function works for a complex case."""
    value_fn, final, ref, expected = complex_test_data

    values = kernel_shapley(value_fn, final, ref, n_samples=10000, seed=42)

    # Check results
    np.testing.assert_allclose(values, expected, atol=0.1)


# Tests for long sequences
def test_long_sequence(long_test_data):
    """Test that all methods work with longer sequences."""
    value_fn, final, ref = long_test_data

    # Get methods and expected runtimes
    methods = [
        (mcmc_shapley, "Monte Carlo"),
        (permutation_sampling_shapley, "Permutation"),
        (kernel_shapley, "Kernel SHAP"),
    ]

    results = {}

    for method_fn, method_name in methods:
        start_time = time.time()

        # Use fewer samples for test speed
        values = method_fn(value_fn, final, ref, n_samples=200, seed=42)

        runtime = time.time() - start_time
        results[method_name] = {"values": values, "runtime": runtime}

        # Check that sum of attributions equals total value difference
        total_diff = (
            value_fn(final.reshape(1, -1))[0] - value_fn(ref.reshape(1, -1))[0]
        )
        np.testing.assert_allclose(values.sum(), total_diff, rtol=0.2)

        # Check patterns in attributions - expect higher values at specific positions
        for i in range(4):
            assert (
                values[i * 4] > values[i * 4 + 2]
            ), f"{method_name}: B should contribute more than D"
            assert (
                values[i * 4 + 1] > values[i * 4 + 3]
            ), f"{method_name}: C should contribute more than E"

    # Print performance comparison
    print(f"\nPerformance for sequence length {len(final)}:")
    for method_name, data in results.items():
        print(f"{method_name}: {data['runtime']:.3f}s")

    # Check consistency between methods
    methods_list = list(results.keys())
    for i, method1 in enumerate(methods_list):
        for method2 in methods_list[i + 1 :]:
            values1 = results[method1]["values"]
            values2 = results[method2]["values"]

            # Compare values using correlation rather than exact matching
            corr = np.corrcoef(values1, values2)[0, 1]
            assert (
                corr > 0.8
            ), f"Correlation between {method1} and {method2} should be high"


# Tests for the unified interface
@pytest.mark.parametrize("method", ["monte_carlo", "permutation", "kernel"])
def test_unified_shapley_interface(simple_test_data, method):
    """Test the unified shapley interface with different methods."""
    value_fn, final, ref, expected = simple_test_data

    values = shapley(
        value_fn, final, ref, method=method, n_samples=500, seed=42
    )

    # Check they produce reasonable values
    np.testing.assert_allclose(values, expected, atol=0.1)


def test_unified_shapley_invalid_method(simple_test_data):
    """Test that the unified interface raises an error for invalid methods."""
    value_fn, final, ref, _ = simple_test_data

    with pytest.raises(ValueError):
        shapley(value_fn, final, ref, method="invalid")


# Test batch size optimization
def test_batch_size_optimization():
    """Test the batch size optimization function."""

    # Use a simple value function for this test
    def simple_batch_fn(x):
        # Simulate computation that scales with batch size
        time.sleep(0.001 * len(x))
        return np.ones(len(x))

    # Find optimal batch size
    optimal_size = optimize_batch_size(
        simple_batch_fn, sequence_length=8, min_batch=8, max_batch=64
    )

    # The optimal size should be within the range
    assert 8 <= optimal_size <= 64, "Optimal batch size out of expected range"

    # Print the result
    print(f"\nOptimal batch size: {optimal_size}")


# Test cortex model compatibility
def test_cortex_model_function_conversion(mock_cortex_model):
    """Test that the Cortex model conversion function works correctly."""
    # Convert the mock model to a value function
    value_function = cortex_model_to_value_function(mock_cortex_model)

    # Create test sequences
    final_seq = np.array(["B", "C", "A", "A", "B", "A", "A", "A"])
    ref_seq = np.array(["A"] * 8)

    # Test the value function
    values = value_function(np.array([final_seq, ref_seq]))

    # Check that we get expected values
    expected = np.array(
        [0.5 * 2 + 0.3 * 1, 0]
    )  # 2 'B's (0.5 each) and 1 'C' (0.3) vs all 'A's (0)
    np.testing.assert_allclose(values, expected, atol=1e-6)

    # Compute Shapley values
    shapley_values = kernel_shapley(
        value_function, final_seq, ref_seq, n_samples=500, seed=42
    )

    # Check positions with specific tokens have appropriate attributions
    for i, token in enumerate(final_seq):
        if token == "B":
            # 'B' positions should have high positive attribution
            assert (
                shapley_values[i] > 0.2
            ), f"B at position {i} should have high attribution"
        elif token == "C":
            # 'C' position should have moderate positive attribution
            assert (
                shapley_values[i] > 0.1
            ), f"C at position {i} should have moderate attribution"
        else:
            # 'A' positions should have low attribution
            assert (
                abs(shapley_values[i]) < 0.1
            ), f"A at position {i} should have low attribution"


# Parametrized test for increasing sequence lengths
@pytest.mark.parametrize("length", [4, 8, 12, 16])
def test_increasing_sequence_lengths(length):
    """Test performance scaling with increasing sequence lengths."""

    # Create a value function
    def value_function(sequences):
        values = np.zeros(len(sequences))
        for i, seq in enumerate(sequences):
            for pos, token in enumerate(seq):
                if token == "B" and pos % 4 == 0:
                    values[i] += 0.5
                elif token == "C" and pos % 4 == 1:
                    values[i] += 0.3
        return values

    # Create sequences of the specified length
    final_seq = np.array(["B", "C", "D", "E"] * (length // 4 + 1))[:length]
    ref_seq = np.array(["A"] * length)

    methods = ["monte_carlo", "permutation", "kernel"]
    times = {}

    for method in methods:
        start_time = time.time()

        # Use fewer samples for longer sequences to save test time
        n_samples = 200 if length <= 12 else 100

        shapley(
            value_function,
            final_seq,
            ref_seq,
            method=method,
            n_samples=n_samples,
            seed=42,
            verbose=False,
        )

        times[method] = time.time() - start_time

    print(f"\nPerformance for sequence length {length}:")
    for method, runtime in times.items():
        print(f"{method}: {runtime:.3f}s")

    # No assertions, just measuring performance


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
