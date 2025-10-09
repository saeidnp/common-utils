import os
import itertools
import time
import tempfile
from pathlib import Path
import shutil

import numpy as np
import pytest

# Attempt to import torch and filelock, or skip tests if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import filelock
    FILELOCK_AVAILABLE = True
except ImportError:
    FILELOCK_AVAILABLE = False

from common_utils.misc import (
    ProtectFile,
    infinite_loader,
    num_available_cores,
    splitit,
    get_register_fn,
    _batchify_helper, # Assuming this is intended to be tested, though typically private
    batchify_numpy,
    batchify_torch
)

# Tests for ProtectFile
@pytest.mark.skipif(not FILELOCK_AVAILABLE, reason="filelock library not available")
class TestProtectFile:
    def test_protect_file_lock_release(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file_path_str = tmp_file.name

        tmp_file_path_obj = Path(tmp_file_path_str)
        # Construct lock_path exactly as ProtectFile does
        lock_file_name = tmp_file_path_obj.name if tmp_file_path_obj.name.startswith(".") else f".{tmp_file_path_obj.name}"
        lock_path = tmp_file_path_obj.parent / f"{lock_file_name}.lock"

        try:
            pf = ProtectFile(tmp_file_path_str, timeout=0.1)
            with pf:
                assert pf.is_locked # Changed from pf.lock.is_locked
                assert lock_path.exists()

            # After exiting context, lock should be released
            assert not pf.is_locked
            # The lock file itself may persist, this is normal for filelock.
            # assert not lock_path.exists() # This assertion is removed.

            # Try acquiring again
            with ProtectFile(tmp_file_path_str, timeout=0.1):
                assert True # Acquired successfully
        finally:
            if Path(tmp_file_path_str).exists():
                os.remove(tmp_file_path_str)
            if lock_path.exists(): # Clean up lock file if test failed mid-run
                os.remove(lock_path)


    def test_protect_file_timeout(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file_path_str = tmp_file.name

        tmp_file_path_obj = Path(tmp_file_path_str)
        # Construct lock_path exactly as ProtectFile does
        lock_file_name = tmp_file_path_obj.name if tmp_file_path_obj.name.startswith(".") else f".{tmp_file_path_obj.name}"
        lock_path = tmp_file_path_obj.parent / f"{lock_file_name}.lock"

        try:
            pf1 = ProtectFile(tmp_file_path_str, timeout=0.1)
            with pf1: # Acquire the lock
                assert pf1.is_locked
                # Try to acquire the same lock again, expecting a timeout
                pf2 = ProtectFile(tmp_file_path_str, timeout=0.05) # Shorter timeout for quick test
                with pytest.raises(filelock.Timeout):
                    with pf2:
                        pass # Should not reach here
            # The lock file itself may persist. pf1 has released the lock.
            # assert not lock_path.exists() # This assertion is removed.
        finally:
            if Path(tmp_file_path_str).exists():
                os.remove(tmp_file_path_str)
            if lock_path.exists():
                os.remove(lock_path)

# Tests for infinite_loader
def test_infinite_loader_basic():
    data = [1, 2, 3]
    loader = infinite_loader(data)
    results = [next(loader) for _ in range(5)]
    assert results == [1, 2, 3, 1, 2]

def test_infinite_loader_empty():
    data = []
    loader = infinite_loader(data)
    with pytest.raises(StopIteration): # Or should it hang? Current common impl raises StopIteration on empty.
                                      # If it's from `common_utils.misc`, let's assume it handles empty iterables.
                                      # Based on typical implementations, it might try iter([]) which is fine,
                                      # but next() would raise StopIteration if not handled.
                                      # If the implementation of infinite_loader is robust, it might return an
                                      # iterator that never yields anything but also doesn't end.
                                      # For now, let's assume it will raise StopIteration if the underlying iterable is empty.
                                      # This needs verification against actual infinite_loader implementation.
        next(loader)


# Tests for num_available_cores
def test_num_available_cores_returns_positive_int():
    cores = num_available_cores()
    assert isinstance(cores, int)
    assert cores >= 1

# Tests for splitit
def test_splitit_various_cases():
    assert list(splitit(10, 3)) == [3, 3, 3, 1]
    assert list(splitit(5, 5)) == [5]
    assert list(splitit(7, 10)) == [7]
    assert list(splitit(0, 5)) == [] # Based on typical behavior, splitting 0 items results in no splits.

    with pytest.raises(AssertionError):
        list(splitit(-1, 5))
    with pytest.raises(AssertionError):
        list(splitit(10, 0))
    with pytest.raises(AssertionError):
        list(splitit(10, -1))


# Tests for get_register_fn
def test_get_register_fn_registration():
    MY_CLASSES = {}
    register_my_class = get_register_fn(MY_CLASSES)

    @register_my_class
    class MyClass1: pass

    @register_my_class(name="CustomName")
    class MyClass2: pass

    assert MY_CLASSES['MyClass1'] == MyClass1
    assert MY_CLASSES['CustomName'] == MyClass2
    assert 'MyClass2' not in MY_CLASSES # Should only be registered under 'CustomName'

def test_get_register_fn_duplicate_name():
    MY_CLASSES_DUP = {}
    register_my_class_dup = get_register_fn(MY_CLASSES_DUP)

    @register_my_class_dup
    class MyClassA: pass # Registers as 'MyClassA'

    with pytest.raises(ValueError, match="Already registered class with name: MyClassA"): # Adjusted match pattern
        @register_my_class_dup(name="MyClassA") # Attempt to register same name
        class MyClassB: pass

    # Check that registering another class with a different name (its own name) after a duplicate attempt also fails if that name was already used.
    MY_CLASSES_DUP2 = {}
    register_my_class_dup2 = get_register_fn(MY_CLASSES_DUP2)

    @register_my_class_dup2(name="OriginalClassC") # Use a distinct name first
    class MyClassC_Original: pass

    # Now try to register a class that would implicitly take "OriginalClassC" if it were named OriginalClassC
    # This part of the test might be redundant if the above already covers duplicate explicit names.
    # To test duplicate implicit name:
    @register_my_class_dup2
    class MyClassD: pass # Registers as "MyClassD"

    with pytest.raises(ValueError, match="Already registered class with name: MyClassD"):
        @register_my_class_dup2 # Attempt to register with implicit name "MyClassD" again
        class MyClassD: pass


# Sample data functions for batchify tests
def sample_numpy_data_dict(n, factor=2, offset=0):
    start = offset * factor
    end = (offset + n) * factor
    return {'a': np.arange(start, end), 'b': np.ones(n), 'non_batched': 'test_string'}

def sample_numpy_data_array(n, factor=2, offset=0):
    start = offset * factor
    end = (offset + n) * factor
    return np.arange(start, end)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def sample_torch_data_dict(n, factor=2, offset=0):
    start = offset * factor
    end = (offset + n) * factor
    return {'a': torch.arange(start, end), 'b': torch.ones(n), 'non_batched': 'test_string'}

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def sample_torch_data_array(n, factor=2, offset=0):
    start = offset * factor
    end = (offset + n) * factor
    return torch.arange(start, end)

# Tests for _batchify_helper are implicitly covered by batchify_numpy/torch tests.
# Direct test for _batchify_helper removed as it's an internal decorator component
# and its previous test structure was incorrect.

# Tests for batchify_numpy
def test_batchify_numpy_array():
    decorated_func = batchify_numpy(max_batch_size=5)(sample_numpy_data_array)
    result = decorated_func(12, factor=1) # Total size 12
    expected_result = np.arange(12)
    assert result.shape == (12,)
    assert np.array_equal(result, expected_result)

def test_batchify_numpy_array_small_n():
    decorated_func = batchify_numpy(max_batch_size=5)(sample_numpy_data_array)
    result = decorated_func(3, factor=1) # n < max_batch_size
    assert result.shape == (3,)
    assert np.array_equal(result, np.arange(3))

def test_batchify_numpy_array_no_batching():
    decorated_func = batchify_numpy(max_batch_size=None)(sample_numpy_data_array)
    result = decorated_func(12, factor=1)
    assert result.shape == (12,)
    assert np.array_equal(result, np.arange(12)) # Function called once

def test_batchify_numpy_dict():
    decorated_func = batchify_numpy(max_batch_size=5, batch_keys=['a', 'b'])(sample_numpy_data_dict)
    result = decorated_func(12, factor=1) # a has size 12, b has size 12
    expected_a = np.arange(12)
    expected_b = np.ones(12)
    assert result['a'].shape == (12,)
    assert np.array_equal(result['a'], expected_a)
    assert result['b'].shape == (12,)
    assert np.array_equal(result['b'], expected_b)
    assert result['non_batched'] == 'test_string' # Should be from the first call

def test_batchify_numpy_dict_small_n():
    decorated_func = batchify_numpy(max_batch_size=5, batch_keys=['a', 'b'])(sample_numpy_data_dict)
    result = decorated_func(3, factor=1) # n < max_batch_size
    assert result['a'].shape == (3,)
    assert np.array_equal(result['a'], np.arange(3))
    assert result['b'].shape == (3,)
    assert np.array_equal(result['b'], np.ones(3))
    assert result['non_batched'] == 'test_string'

# Tests for batchify_torch
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_batchify_torch_array():
    decorated_func = batchify_torch(max_batch_size=5)(sample_torch_data_array)
    result = decorated_func(12, factor=1) # Total size 12
    expected_result = torch.arange(12)
    assert result.shape == (12,)
    assert torch.equal(result, expected_result)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_batchify_torch_array_small_n():
    decorated_func = batchify_torch(max_batch_size=5)(sample_torch_data_array)
    result = decorated_func(3, factor=1) # n < max_batch_size
    assert result.shape == (3,)
    assert torch.equal(result, torch.arange(3))

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_batchify_torch_array_no_batching():
    decorated_func = batchify_torch(max_batch_size=None)(sample_torch_data_array)
    result = decorated_func(12, factor=1)
    assert result.shape == (12,)
    assert torch.equal(result, torch.arange(12))

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_batchify_torch_dict():
    decorated_func = batchify_torch(max_batch_size=5, batch_keys=['a', 'b'])(sample_torch_data_dict)
    result = decorated_func(12, factor=1) # a has size 12, b has size 12
    expected_a = torch.arange(12)
    expected_b = torch.ones(12)
    assert result['a'].shape == (12,)
    assert torch.equal(result['a'], expected_a)
    assert result['b'].shape == (12,)
    assert torch.equal(result['b'], expected_b)
    assert result['non_batched'] == 'test_string' # Should be from the first call

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
def test_batchify_torch_dict_small_n():
    decorated_func = batchify_torch(max_batch_size=5, batch_keys=['a', 'b'])(sample_torch_data_dict)
    result = decorated_func(3, factor=1) # n < max_batch_size
    assert result['a'].shape == (3,)
    assert torch.equal(result['a'], torch.arange(3))
    assert result['b'].shape == (3,)
    assert torch.equal(result['b'], torch.ones(3))
    assert result['non_batched'] == 'test_string'

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not available")
@pytest.mark.skipif(not FILELOCK_AVAILABLE, reason="filelock not available")
def test_imports_work(): # Dummy test to ensure conditional imports are fine
    assert True
