import random
import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None # Placeholder if torch is not available

from common_utils.random import (
    set_random_seed,
    get_random_state,
    set_random_state,
    RNG,
    rng_decorator
)

# Helper function to generate random numbers from different sources
def generate_random_numbers(num_samples):
    numbers = {
        'python': [random.random() for _ in range(num_samples)],
        'numpy': np.random.rand(num_samples)
    }
    if TORCH_AVAILABLE:
        numbers['torch'] = torch.rand(num_samples)
        # We are not generating cuda numbers here, focusing on CPU.
        # common_utils.random handles torch.cuda state internally.
    return numbers

# Test set_random_seed
def test_set_random_seed():
    set_random_seed(42)
    numbers1 = generate_random_numbers(5)

    set_random_seed(42) # Reset seed
    numbers2 = generate_random_numbers(5)

    assert numbers1['python'] == numbers2['python'], "Python random numbers should match after resetting seed"
    assert np.array_equal(numbers1['numpy'], numbers2['numpy']), "NumPy random numbers should match after resetting seed"

    if TORCH_AVAILABLE:
        assert torch.equal(numbers1['torch'], numbers2['torch']), "PyTorch random numbers should match after resetting seed"

# Test get_random_state and set_random_state
def test_get_set_random_state():
    set_random_seed(10)
    state1 = get_random_state()
    numbers1_part1 = generate_random_numbers(3)

    # Change state and advance RNGs
    set_random_seed(20)
    _ = generate_random_numbers(3) # Discard these numbers

    # Restore original state
    set_random_state(state1)
    numbers1_part2_restored = generate_random_numbers(3) # Should continue from where state1 left off
                                                       # which is effectively generating the same numbers as numbers1_part1
                                                       # because generate_random_numbers itself doesn't advance the state before generating.

    # This test logic is a bit tricky. get_random_state captures the state *before* numbers are generated.
    # So if we generate numbers1_part1, then capture state, then restore state, then generate numbers again,
    # the second set of numbers should be the *same* as numbers1_part1.
    # Let's refine:
    set_random_seed(10)
    numbers_initial_state_check = generate_random_numbers(3) # Generate reference numbers

    set_random_seed(10) # Reset to make sure we are at the same point
    state_before_gen = get_random_state() # Get state
    numbers_part1 = generate_random_numbers(3) # Generate first part of sequence

    assert numbers_initial_state_check['python'] == numbers_part1['python']

    # Mess up the state
    set_random_seed(99)
    _ = generate_random_numbers(10)

    # Restore state
    set_random_state(state_before_gen)
    numbers_part1_restored = generate_random_numbers(3) # Should be same as numbers_part1

    assert numbers_part1['python'] == numbers_part1_restored['python'], "Python numbers differ after state restoration"
    assert np.array_equal(numbers_part1['numpy'], numbers_part1_restored['numpy']), "Numpy numbers differ after state restoration"
    if TORCH_AVAILABLE:
        assert torch.equal(numbers_part1['torch'], numbers_part1_restored['torch']), "Torch numbers differ after state restoration"

    # Now, let's test continuing the sequence
    set_random_seed(10) # Start sequence
    full_sequence_ref_py = [random.random() for _ in range(6)]
    full_sequence_ref_np = np.random.rand(6)
    if TORCH_AVAILABLE:
        full_sequence_ref_torch = torch.rand(6)

    set_random_seed(10) # Reset for test
    _ = generate_random_numbers(3) # Generate first 3 (these are numbers_part1)
    state_after_part1 = get_random_state() # Capture state after generating 3 numbers

    # Mess up state
    set_random_seed(88)
    _ = generate_random_numbers(5)

    set_random_state(state_after_part1) # Restore state to after part1 was generated
    numbers_part2 = generate_random_numbers(3) # Generate next 3 numbers

    assert full_sequence_ref_py[3:] == numbers_part2['python']
    assert np.array_equal(full_sequence_ref_np[3:], numbers_part2['numpy'])
    if TORCH_AVAILABLE:
        assert torch.equal(full_sequence_ref_torch[3:], numbers_part2['torch'])


# Test RNG context manager
class TestRNGContext:
    def test_rng_context_isolation(self):
        set_random_seed(100)
        outer_numbers_before_py = [random.random() for _ in range(3)]
        outer_numbers_before_np = np.random.rand(3)
        if TORCH_AVAILABLE:
            outer_numbers_before_torch = torch.rand(3)

        with RNG(seed=200):
            inner_numbers1 = generate_random_numbers(3)

        outer_numbers_after_py = [random.random() for _ in range(3)]
        outer_numbers_after_np = np.random.rand(3)
        if TORCH_AVAILABLE:
            outer_numbers_after_torch = torch.rand(3)

        # Check that outer RNG state was affected by its own generation, not by inner
        assert outer_numbers_before_py != outer_numbers_after_py
        assert not np.array_equal(outer_numbers_before_np, outer_numbers_after_np)
        if TORCH_AVAILABLE:
            assert not torch.equal(outer_numbers_before_torch, outer_numbers_after_torch)

        # Check reproducibility of inner context
        with RNG(seed=200): # Re-enter with same seed
            inner_numbers2 = generate_random_numbers(3)

        assert inner_numbers1['python'] == inner_numbers2['python']
        assert np.array_equal(inner_numbers1['numpy'], inner_numbers2['numpy'])
        if TORCH_AVAILABLE:
            assert torch.equal(inner_numbers1['torch'], inner_numbers2['torch'])

    def test_rng_context_state_restoration(self):
        set_random_seed(300)
        expected_sequence_py = [random.random() for _ in range(6)]
        expected_sequence_np = np.random.rand(6)
        if TORCH_AVAILABLE:
            expected_sequence_torch = torch.rand(6)

        set_random_seed(300) # Reset for the test execution

        sequence_part1_py = [random.random() for _ in range(3)]
        sequence_part1_np = np.random.rand(3)
        if TORCH_AVAILABLE:
            sequence_part1_torch = torch.rand(3)

        with RNG(seed=999): # This seed is for the context
            # Generate some numbers inside context to use its RNG
            _ = generate_random_numbers(100)

        sequence_part2_py = [random.random() for _ in range(3)]
        sequence_part2_np = np.random.rand(3)
        if TORCH_AVAILABLE:
            sequence_part2_torch = torch.rand(3)

        assert sequence_part1_py + sequence_part2_py == expected_sequence_py
        assert np.array_equal(np.concatenate((sequence_part1_np, sequence_part2_np)), expected_sequence_np)
        if TORCH_AVAILABLE:
            assert torch.equal(torch.cat((sequence_part1_torch, sequence_part2_torch)), expected_sequence_torch)


# Test rng_decorator
def test_rng_decorator_isolation_and_reproducibility():
    @rng_decorator(seed=400)
    def decorated_func_A():
        return generate_random_numbers(3)

    @rng_decorator(seed=400) # Same seed
    def decorated_func_B():
        return generate_random_numbers(3)

    # Test that external state does not affect decorated function
    set_random_seed(10)
    res1_A = decorated_func_A()

    set_random_seed(20) # Change external seed
    res2_A = decorated_func_A() # Call first function again

    # Results from same decorated function (with fixed seed) should be identical despite external changes
    assert res1_A['python'] == res2_A['python']
    assert np.array_equal(res1_A['numpy'], res2_A['numpy'])
    if TORCH_AVAILABLE:
        assert torch.equal(res1_A['torch'], res2_A['torch'])

    # Results from different decorated functions with THE SAME seed should also be identical
    res1_B = decorated_func_B()
    assert res1_A['python'] == res1_B['python']
    assert np.array_equal(res1_A['numpy'], res1_B['numpy'])
    if TORCH_AVAILABLE:
        assert torch.equal(res1_A['torch'], res1_B['torch'])


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
def test_torch_specific_behaviors_if_needed():
    # This is a placeholder if we need to test anything specific to torch's RNG
    # beyond what generate_random_numbers and the main functions cover.
    # For example, direct seeding of torch.cuda.manual_seed_all if common_utils.random
    # has specific logic for it that needs isolated testing.
    # Current tests rely on set_random_seed and state functions to handle torch state.
    assert TORCH_AVAILABLE # Should only run if torch is available.

    # Example: Check if set_random_seed affects torch.cuda if available
    if hasattr(torch, 'cuda') and torch.cuda.is_available():
        set_random_seed(12345)
        cuda_rand1 = torch.cuda.FloatTensor(5).normal_() # Generate some CUDA random numbers
        set_random_seed(12345)
        cuda_rand2 = torch.cuda.FloatTensor(5).normal_()
        assert torch.equal(cuda_rand1, cuda_rand2), "Torch CUDA numbers should match after resetting seed"
    else:
        pytest.skip("PyTorch CUDA not available")

# It's good practice to also test the case where TORCH_AVAILABLE is False for some functions
# to ensure they don't error out, but the skips should handle this.
# The common_utils.random module itself should be robust to torch not being there if it's optional.
# Our current generate_random_numbers helper and skips manage this at the test level.
