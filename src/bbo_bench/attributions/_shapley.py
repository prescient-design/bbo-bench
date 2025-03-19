import itertools
import math
import time
from typing import Callable, Optional

import numpy as np


def get_shapley_weight(vector, index, N):
    """Compute the weight for the Shapley values."""
    S = int(N - np.sum(vector) if vector[index] else N - np.sum(vector) - 1)
    return 1.0 / (math.comb(N - 1, S))


def _check_reshape_inputs(example_seq, ref_seq):
    """Check and potentially reshape inputs for the Shapley computation.

    Args:
        example_seq: The example sequence to compute Shapley values for.
        ref_seq: The reference sequence to compare against.
    """
    # Check shape constraints
    assert (
        example_seq.shape == ref_seq.shape
    ), "example_seq and ref_seq should have the same shape"
    assert (
        len(example_seq.shape) == 1 or example_seq.shape[0] == 1
    ), "example_seq hould be 1D or nominally 2D"
    # Reshape if necessary
    if len(example_seq.shape) > 1:
        example_seq = example_seq[0]
        ref_seq = ref_seq[0]
    assert len(example_seq.shape) == 1, "example_seq should be 1D"
    assert len(example_seq) == len(
        ref_seq
    ), "example_seq and ref_seq should have the same length"
    return example_seq, ref_seq


def exact_shapley(
    value_function,
    example_seq: np.ndarray,
    ref_seq: np.ndarray,
):
    """Compute Shapley values for the given model and input x.

    Args:
        model: The model to compute Shapley values for.
        x: The input data.

    Returns:
        A list of Shapley values for each feature in x.
    """
    example_seq, ref_seq = _check_reshape_inputs(example_seq, ref_seq)

    N = len(example_seq)

    # Initialize Shapley values
    shap_values = np.zeros(N)

    # Compute power set of features either taking value design_i or reference_i
    power_set = np.array(list(itertools.product([0, 1], repeat=N)))
    mut_power_set = np.where(power_set, example_seq, ref_seq)
    # print(mut_power_set)

    # Compute the rewards for each feature
    rewards = value_function(mut_power_set).flatten()
    rewards[rewards == -np.inf] = 0
    # print("Values:", rewards)

    for index in range(N):
        # Compute the weights for the Shapley values
        weights = np.array(
            [get_shapley_weight(vector, index, N) for vector in power_set]
        )
        sign = np.array([1 if vector[index] else -1 for vector in power_set])

        # Compute the Shapley values
        shap_values[index] = np.sum(weights * rewards * sign) / N

    # print("Shapley:", shap_values)
    return shap_values


def mcmc_shapley(
    value_function: Callable,
    example_seq: np.ndarray,
    ref_seq: np.ndarray,
    n_samples: int = 1000,
    batch_size: int = 128,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute approximate Shapley values using Monte Carlo sampling.

    This function implements a sampling-based approach to computing Shapley values,
    which makes it feasible for longer sequences.

    Args:
        value_function: Function that takes a batch of sequences and returns their values.
        example_seq: Target sequence for which to compute attributions.
        ref_seq: Reference sequence to compare against.
        n_samples: Number of permutation samples to use for approximation.
        batch_size: Batch size for evaluating the value function.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress information.

    Returns:
        Array of Shapley values for each position in the sequence.
    """
    # Validate inputs
    example_seq, ref_seq = _check_reshape_inputs(example_seq, ref_seq)

    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    N = len(example_seq)
    shap_values = np.zeros(N)

    if verbose:
        print(f"Computing Shapley values for sequence of length {N}")
        print(f"Using {n_samples} Monte Carlo samples")
        start_time = time.time()

    # Generate n_samples random permutations
    permutations = np.array(
        [np.random.permutation(N) for _ in range(n_samples)]
    )

    # Process in batches to avoid memory issues
    n_batches = int(np.ceil(n_samples / batch_size))

    for batch_idx in range(n_batches):
        if verbose and batch_idx % max(1, n_batches // 10) == 0:
            print(f"Processing batch {batch_idx+1}/{n_batches}...")

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)
        batch_size_actual = end_idx - start_idx

        batch_perms = permutations[start_idx:end_idx]

        # For each feature position, compute marginal contributions
        for pos in range(N):
            # For each permutation in the batch
            for perm_idx, perm in enumerate(batch_perms):
                # Find position of current feature in this permutation
                feature_pos_in_perm = np.where(perm == pos)[0][0]

                # Create two sequences: one without the feature and one with it
                seq_without = np.copy(ref_seq)
                seq_with = np.copy(ref_seq)

                # Add features that come before current feature in the permutation
                for j in range(feature_pos_in_perm):
                    prev_feature = perm[j]
                    seq_without[prev_feature] = example_seq[prev_feature]
                    seq_with[prev_feature] = example_seq[prev_feature]

                # Add the current feature only to seq_with
                seq_with[pos] = example_seq[pos]

                # Evaluate both sequences
                if perm_idx == 0:
                    # Initialize arrays for the batch
                    seqs_without = np.zeros(
                        (batch_size_actual, N), dtype=example_seq.dtype
                    )
                    seqs_with = np.zeros(
                        (batch_size_actual, N), dtype=example_seq.dtype
                    )

                seqs_without[perm_idx] = seq_without
                seqs_with[perm_idx] = seq_with

            # Evaluate the value function on the batches
            values_without = value_function(seqs_without)
            values_with = value_function(seqs_with)

            # Update Shapley values with the marginal contributions
            marginal_contributions = values_with - values_without
            shap_values[pos] += np.sum(marginal_contributions)

    # Average the contributions
    shap_values /= n_samples

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"Shapley computation completed in {elapsed_time:.2f} seconds")

    return shap_values


def optimize_batch_size(
    value_function: Callable,
    sequence_length: int,
    dtype: type = np.str_,
    min_batch: int = 16,
    max_batch: int = 1024,
) -> int:
    """Find an optimal batch size for the value function.

    Args:
        value_function: The value function to optimize for.
        sequence_length: Length of sequences to test with.
        dtype: Data type of the sequences.
        min_batch: Minimum batch size to try.
        max_batch: Maximum batch size to try.

    Returns:
        Recommended batch size.
    """
    # Generate random test data
    test_data = np.random.choice(
        ["A", "B", "C", "D"], size=(max_batch, sequence_length)
    ).astype(dtype)

    batch_sizes = [min_batch]
    while batch_sizes[-1] * 2 <= max_batch:
        batch_sizes.append(batch_sizes[-1] * 2)

    times = []

    print("Finding optimal batch size...")
    for bs in batch_sizes:
        start_time = time.time()
        # Warm up
        _ = value_function(test_data[:bs])

        # Timing run
        start_time = time.time()
        _ = value_function(test_data[:bs])
        elapsed = time.time() - start_time

        times.append(elapsed / bs)  # Time per sample
        print(
            f"Batch size {bs}: {elapsed:.5f}s total, {times[-1]:.7f}s per sample"
        )

    # Find the batch size with the minimum time per sample
    best_idx = np.argmin(times)
    return batch_sizes[best_idx]


def permutation_sampling_shapley(
    value_function: Callable,
    example_seq: np.ndarray,
    ref_seq: np.ndarray,
    n_samples: int = 200,
    batch_size: int = 32,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute Shapley values using permutation sampling.

    This is an alternative implementation that processes entire permutations at once,
    which can be more efficient for some value functions.

    Args:
        value_function: Function that takes a batch of sequences and returns their values.
        example_seq: Target sequence for which to compute attributions.
        ref_seq: Reference sequence to compare against.
        n_permutations: Number of permutations to sample.
        batch_size: Number of sequences to evaluate at once.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress.

    Returns:
        Array of Shapley values for each position in the sequence.
    """
    # Input validation
    example_seq, ref_seq = _check_reshape_inputs(example_seq, ref_seq)

    if seed is not None:
        np.random.seed(seed)

    N = len(example_seq)
    shap_values = np.zeros(N)

    if verbose:
        print("Computing Shapley values using permutation sampling")
        print(f"Sequence length: {N}, Permutations: {n_samples}")
        start_time = time.time()

    # Generate permutations
    permutations = [np.random.permutation(N) for _ in range(n_samples)]

    # For each permutation
    for p_idx, perm in enumerate(permutations):
        if verbose and (p_idx + 1) % 10 == 0:
            print(f"Processing permutation {p_idx+1}/{n_samples}")

        # Start with the reference sequence
        current_seq = ref_seq.copy()
        prev_value = value_function(current_seq.reshape(1, -1))[0]

        # Add features one by one according to the permutation
        for pos_idx, pos in enumerate(perm):
            # Update the sequence with the feature from example_seq
            current_seq[pos] = example_seq[pos]

            # Evaluate the new sequence
            new_value = value_function(current_seq.reshape(1, -1))[0]

            # Update Shapley values with the marginal contribution
            shap_values[pos] += new_value - prev_value

            # Update prev_value for the next iteration
            prev_value = new_value

    # Average the contributions
    shap_values /= n_samples

    if verbose:
        elapsed_time = time.time() - start_time
        print(f"Completed in {elapsed_time:.2f} seconds")

    return shap_values


def kernel_shapley(
    value_function: Callable,
    example_seq: np.ndarray,
    ref_seq: np.ndarray,
    n_samples: int = 2048,
    reg_param: float = 0.01,
    batch_size: int = 128,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Compute Shapley values using the Kernel SHAP algorithm.

    This implements a variant of the Kernel SHAP algorithm, which is especially
    efficient for longer sequences.

    Args:
        value_function: Function that takes a batch of sequences and returns their values.
        example_seq: Target sequence for which to compute attributions.
        ref_seq: Reference sequence to compare against.
        n_samples: Number of samples to use for approximation.
        reg_param: Regularization parameter for the weighted linear regression.
        batch_size: Batch size for evaluating the value function.
        seed: Random seed for reproducibility.
        verbose: Whether to print progress information.

    Returns:
        Array of Shapley values for each position in the sequence.
    """
    # Validate inputs
    example_seq, ref_seq = _check_reshape_inputs(example_seq, ref_seq)

    if seed is not None:
        np.random.seed(seed)

    N = len(example_seq)

    if verbose:
        print(f"Computing Kernel SHAP values for sequence of length {N}")
        print(f"Using {n_samples} samples")
        start_time = time.time()

    # Ensure n_samples is large enough
    n_samples = max(n_samples, N + 1)

    # Sample coalitions (binary masks indicating which features to include)
    if N <= 15:
        # For small N, we can include all combinations with 0 and N features
        coalition_sizes = np.random.choice(
            np.arange(1, N), size=n_samples - 2, replace=True
        )
        coalition_sizes = np.concatenate([coalition_sizes, [0, N]])
    else:
        # For larger N, sample coalition sizes according to a weighting
        coalition_sizes = np.random.choice(
            np.arange(1, N),
            size=n_samples - 2,
            p=np.array([1 / (i * (N - i)) for i in range(1, N)])
            / sum(1 / (i * (N - i)) for i in range(1, N)),
            replace=True,
        )
        coalition_sizes = np.concatenate([coalition_sizes, [0, N]])

    # Create coalitions (binary masks)
    coalitions = []
    for size in coalition_sizes:
        if size == 0:
            # Empty coalition
            coalitions.append(np.zeros(N, dtype=bool))
        elif size == N:
            # Full coalition
            coalitions.append(np.ones(N, dtype=bool))
        else:
            # Random coalition of specified size
            coalition = np.zeros(N, dtype=bool)
            coalition[
                np.random.choice(N, size=int(size), replace=False)
            ] = True
            coalitions.append(coalition)

    coalitions = np.array(coalitions)

    # Create sequences based on the coalitions
    sequences = np.zeros((n_samples, N), dtype=example_seq.dtype)
    for i, coalition in enumerate(coalitions):
        sequences[i] = np.where(coalition, example_seq, ref_seq)

    # Evaluate sequences in batches
    n_batches = int(np.ceil(n_samples / batch_size))
    values = np.zeros(n_samples)

    for batch_idx in range(n_batches):
        if verbose and batch_idx % max(1, n_batches // 10) == 0:
            print(f"Evaluating batch {batch_idx+1}/{n_batches}...")

        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_samples)

        batch_sequences = sequences[start_idx:end_idx]
        batch_values = value_function(batch_sequences)

        values[start_idx:end_idx] = batch_values

    # Compute Shapley kernel weights
    coalition_sizes = coalitions.sum(axis=1)
    weights = np.array(
        [(N - 1) / (k * (N - k)) if 0 < k < N else 1 for k in coalition_sizes]
    )

    # Handle potential division by zero
    weights = np.where(np.isfinite(weights), weights, 0)

    # Perform weighted linear regression to compute Shapley values
    X = coalitions.astype(float)
    y = values

    # Add L2 regularization to handle potential collinearity
    XTX = X.T @ (X * weights[:, np.newaxis])
    XTX += reg_param * np.eye(N)
    XTy = X.T @ (y * weights)

    shap_values = np.linalg.solve(XTX, XTy)

    if verbose:
        elapsed_time = time.time() - start_time
        print(
            f"Kernel SHAP computation completed in {elapsed_time:.2f} seconds"
        )

    return shap_values


def cortex_model_to_value_function(model):
    """Convert a cortex model to a value function compatible with Shapley calculation.

    Args:
        model: A cortex model that can process string arrays.

    Returns:
        A function that takes a batch of sequences and returns their values.
    """

    def inner_model_value_function(seqs):
        # Format sequences for cortex by adding spaces between tokens
        str_array = np.array([" ".join(seq) for seq in seqs])
        tree_output = model.call_from_str_array(
            str_array=str_array, corrupt_frac=0.0
        )
        mean_pred = (
            tree_output.fetch_task_outputs("generic_task")["loc"]
            .squeeze(-1)
            .mean(0)
        )
        return mean_pred.detach().cpu().numpy()

    return inner_model_value_function


def shapley(
    value_function: Callable,
    example_seq: np.ndarray,
    ref_seq: np.ndarray,
    method: str = "exact",
    **kwargs,
) -> np.ndarray:
    """Compute Shapley values using the specified method.

    Args:
        value_function: Function that takes a batch of sequences and returns their values.
        example_seq: Target sequence for which to compute attributions.
        ref_seq: Reference sequence to compare against.
        method: Method to use for computing Shapley values:
            - 'exact': Exact computation (only for short sequences)
            - 'monte_carlo': Monte Carlo sampling
            - 'permutation': Permutation sampling
            - 'kernel': Kernel SHAP algorithm (default, recommended for longer sequences)
        **kwargs: Additional arguments to pass to the specific method.

    Returns:
        Array of Shapley values for each position in the sequence.
    """
    if method == "exact":
        # Import the exact computation function from the existing code
        # from ._shapley import shapley as exact_shapley
        return exact_shapley(value_function, example_seq, ref_seq)
    elif method == "monte_carlo":
        return mcmc_shapley(value_function, example_seq, ref_seq, **kwargs)
    elif method == "permutation":
        return permutation_sampling_shapley(
            value_function, example_seq, ref_seq, **kwargs
        )
    elif method == "kernel":
        return kernel_shapley(value_function, example_seq, ref_seq, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. Available methods: 'exact', 'monte_carlo', 'permutation', 'kernel'."
        )
