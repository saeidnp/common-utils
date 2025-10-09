# Common Utils

[![CI](https://github.com/saeidnp/common-utils/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saeidnp/common-utils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/saeidnp/common-utils/graph/badge.svg?token=JLQ6Q6BDD5)](https://codecov.io/gh/saeidnp/common-utils)

`common-utils` is a lightweight Python library that provides a collection of small, reusable utilities to streamline your projects. It includes helpers for logging, reproducible randomness, plotting, file protection, and PyTorch-specific tasks.

## Features

- **Lightweight:** The package is designed to be minimal and have few dependencies.
- **Easy to use:** The utilities are simple to integrate into any Python project.
- **Well-tested:** The library has a comprehensive test suite to ensure reliability.

## Installation

You can install `common-utils` from PyPI or directly from this repository.

### From PyPI

```bash
pip install common-utils
```

### From GitHub

```bash
pip install "git+https://github.com/saeidnp/common-utils.git@main#egg=common_utils"
```

For development, you can clone the repository and install it in editable mode:

```bash
git clone https://github.com/saeidnp/common-utils.git
cd common-utils
pip install -e .
```

## Package Overview

The library is organized into the following modules:

- **`common_utils.const`**: Defines common constants, such as `TMPDIR`.
- **`common_utils.logging`**: Provides lightweight helpers for `wandb` and a `LoggingHandler` class.
- **`common_utils.misc`**: Includes utilities for file locks, batching, and other conveniences.
- **`common_utils.plotting`**: Contains helpers for creating plots and handling colors.
- **`common_utils.random`**: Offers tools for deterministic random number generation across Python, NumPy, and PyTorch.
- **`common_utils.ptutils`**: A subpackage with PyTorch-related utilities.

## Usage

Here are a few examples of how to use `common-utils` in your projects.

### Reproducible Randomness

To ensure your code is deterministic, you can set a global random seed:

```python
from common_utils.random import set_random_seed, RNG

# Set a global seed for reproducibility
set_random_seed(42)

# Create a deterministic section of code
with RNG(42):
    # Your code here will be deterministic
    pass
```

### File Protection

You can protect a file from concurrent access using a lock:

```python
from common_utils.misc import ProtectFile

with ProtectFile('/tmp/my_file.txt'):
    # Safely read or write to the file
    pass
```

### Logging with `wandb`

The library provides a simple way to initialize `wandb`:

```python
from common_utils.logging import init

# Requires WANDB_PROJECT and WANDB_ENTITY env vars
# or you can pass them explicitly
run = init(config={'lr': 1e-3}, project='my-project', entity='my-entity')
```

### Plotting Helpers

You can easily set the size of your plots for consistent styling:

```python
from common_utils.plotting import set_size, colorblind_cycle

# Set the plot size for a thesis document
size = set_size('thesis', fraction=0.5)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
