# Common Utils

[![CI](https://github.com/saeidnp/common-utils/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saeidnp/common-utils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/saeidnp/common-utils/branch/main/graph/badge.svg?token=JLQ6Q6BDD5)](https://codecov.io/gh/saeidnp/common-utils)

`common-utils` is a lightweight Python library that provides a collection of small, reusable utilities to streamline your projects. It includes helpers for logging, reproducible randomness, plotting, file protection, and PyTorch-specific tasks.

## Installation

You can install `common-utils` directly from this repository.

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

- **`common_utils.random`**: Offers tools for deterministic random number generation across Python, NumPy, and PyTorch.
- **`common_utils.logging`**: Provides lightweight helpers for `wandb` and a `LoggingHandler` class.
- **`common_utils.plotting`**: Contains helpers for creating publication-ready plots (mainly correct plot figure sizes and LaTeX-rendered text) and handling colors.
- **`common_utils.ptutils`**: A subpackage with PyTorch-related utilities.
- **`common_utils.misc`**: Includes utilities for file locks, batching, and other conveniences.

## Usage

Here are a few examples of how to use `common-utils` in your projects.

### Reproducible Randomness

To ensure your code is deterministic, you can set a global random seed:

```python
from common_utils.random import set_random_seed, RNG, rng_decorator

# Set a global seed for reproducibility
set_random_seed(42)

# Create a deterministic section of code
with RNG(42):
    # Your code here will be deterministic
    pass

# Create a function with a fixed seed for reproducability and progress tracking
@rng_decorator(42)
def generate_samples(model, batch_size):
    return model.generate(batch_size)
```

### File Protection

You can protect a file from concurrent access using a lock:

```python
from common_utils.misc import ProtectFile

with ProtectFile('my_file.txt'):
    # Safely read or write to the file
    # It makes a lock file with path derived from the filename. It acquires the lock in this block and frees it upon exitting
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
The difference between `common_utils.logging.init` and a simple `wandb.init` is that the init function here:
- Includes the starting python file name as a tag.
- Includes the Slurm job id in the job notes (only if the environment variable `_MY_JOB_ID` is set)

Furthermore, `common_utils.logging.support_unobserve` allows passing `--unobserve` to your python script to locally put wandb in "offline" mode. This feature mimic's [Sacred](https://github.com/IDSIA/sacred)'s behaviour for skipping tracking of jobs.

### Plotting Helpers

You can easily set the size of your plots for consistent styling:

```python
from common_utils.plotting import set_size

# Set the plot size for a thesis document
size = set_size(page_width, fraction=0.5)
```
See the source code for links to how to set the `page_width` extracted from your LaTeX document.

Moreover, you can simply call `common_utils.plotting.setup_matplotlib` right after importing matplotlib to setup the defaults to use LaTeX rendering with font and font sizes compatible with most papers.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
