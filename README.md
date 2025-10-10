# Common Utils

[![CI](https://github.com/saeidnp/common-utils/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/saeidnp/common-utils/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/saeidnp/common-utils/branch/main/graph/badge.svg?token=JLQ6Q6BDD5)](https://codecov.io/gh/saeidnp/common-utils)

`common-utils` is a lightweight Python library containing small, reusable utilities that I commonly used in my machine learning projects. It includes helpers for logging, reproducible randomness, plotting, file protection, and PyTorch-specific tasks.

## Installation

Install directly from this repository:

```bash
pip install "git+https://github.com/saeidnp/common-utils.git@main#egg=common_utils"
```

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/saeidnp/common-utils.git
cd common-utils
pip install -e .
```

## Package Overview

The library is organized into the following modules:

- `common_utils.random`: Offers tools for deterministic random number generation across Python, NumPy, and PyTorch.
- `common_utils.logging`: Provides lightweight helpers for `wandb` and a `LoggingHandler` class.
- `common_utils.plotting`: Helpers for publication-ready plots (figure sizes, LaTeX rendering) and color utilities.
- `common_utils.ptutils`: PyTorch-related utilities.
- `common_utils.misc`: Utilities for file locks, batching, and other conveniences.

## Usage

Some short examples demonstrating common usage patterns.

### Reproducible Randomness

To ensure your code is deterministic, you can set a global or local random seed:

```python
from common_utils.random import set_random_seed, RNG, rng_decorator

# Set the global random seed
set_random_seed(42)

# Set the random seed locally in a block
with RNG(42):
    # Code inside this context will be deterministic
    pass

# Make a deterministic function by setting its random seed locally every time it is called
@rng_decorator(42)
def generate_samples(model, batch_size):
    return model.generate(batch_size)
```

### File protection

Protect a file from concurrent access using a lock:

```python
from common_utils.misc import ProtectFile

with ProtectFile('my_file.txt'):
    # Safely read from or write to the file
    pass
```

### Logging with wandb

Initialize `wandb` using the helper function:

```python
from common_utils.logging import init

# Requires WANDB_PROJECT and WANDB_ENTITY environment variables, or pass them explicitly
run = init(config={'lr': 1e-3}, project='my-project', entity='my-entity')
```

Differences compared to `wandb.init`:

- The script name is added as a tag.
- The Slurm job id is included in the run notes.

Use `common_utils.logging.support_unobserve()` in scripts to allow a `--unobserve` flag which sets `WANDB_MODE=dryrun` (useful for running locally without logging to wandb). This behavior mirrors Sacred's offline behaviour when skipping run tracking.

### Plotting helpers

Set figure sizes that match LaTeX documents:

```python
from common_utils.plotting import set_size

# page_width can be a numeric point value or one of the presets like 'thesis'
size = set_size(page_width, fraction=0.5)
```

Call `common_utils.plotting.setup_matplotlib()` to configure matplotlib for LaTeX rendering and consistent font sizes suitable for publications.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
