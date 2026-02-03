import wandb
import os, sys
import numpy as np
import time


def support_unobserve():
    """
    Check for `--unobserve` in sys.argv and set WANDB_MODE to offline if present.

    Removes `--unobserve` from `sys.argv` to clean up arguments.
    """
    if "--unobserve" in sys.argv:
        sys.argv.remove("--unobserve")
        os.environ["WANDB_MODE"] = "offline"


def init(config, project=None, entity=None, tags=None, notes=None, **kwargs):
    """
    Initialize Weights & Biases logging.

    Args:
        config (dict): Configuration dictionary to log.
        project (str, optional): Project name. Defaults to None.
        entity (str, optional): Entity name. Defaults to None.
        tags (list, optional): List of tags. Defaults to None.
        notes (str, optional): Notes for the run. Defaults to None.
        **kwargs: Additional arguments passed to `wandb.init`.

    Returns:
        wandb.run: The wandb run object.
    """
    if tags is None:
        tags = []
    if entity is None:
        assert (
            "WANDB_ENTITY" in os.environ
        ), "Please either pass in \"entity\" to logging.init or set environment variable 'WANDB_ENTITY' to your wandb entity name."
    if project is None:
        assert (
            "WANDB_PROJECT" in os.environ
        ), "Please either pass in \"project\" to logging.init or set environment variable 'WANDB_PROJECT' to your wandb project name."
    tags.append(os.path.basename(sys.argv[0]))
    if "SLURM_JOB_ID" in os.environ:
        x = f"(jobid:{os.environ['SLURM_JOB_ID']})"
        notes = x if notes is None else notes + " " + x
    if "SLURM_ARRAY_JOB_ID" in os.environ:
        array_job_id = os.getenv("SLURM_ARRAY_JOB_ID")
        array_task_id = os.getenv("SLURM_ARRAY_TASK_ID")
        x = f"(job_array_id:{array_job_id}_{array_task_id})"
        notes = x if notes is None else notes + " " + x
    return wandb.init(
        project=project,
        entity=entity,
        config=config,
        tags=tags,
        notes=notes,
        **kwargs,
    )


class LoggingHandler:
    """
    A helper class to accumulate logs and flash them periodically.
    """
    def __init__(self):
        """Initialize the LoggingHandler."""
        self.log_count = 0
        self.reset()

    def log(self, kwargs):
        """
        Accumulate logs.

        Args:
            kwargs (dict): Key-value pairs to log.
        """
        assert (
            "between_log_time" not in kwargs
        ), "Please do not use 'between_log_time' as a key in your logging dictionary."
        self.log_count += 1
        if self.log_dict == {}:
            self.log_dict = {k: [] for k, v in kwargs.items()}
            self.log_dict["between_log_time"] = []
            self.t_0 = time.time()
        else:
            kwargs["between_log_time"] = time.time() - self.t_0
            self.t_0 = time.time()

        for k, v in kwargs.items():
            if k not in self.log_dict:
                self.log_dict[k] = []
                # raise Exception(f"Key {k} not in log_dict. Keys are {self.log_dict.keys()}")
            self.log_dict[k].append(v)

    def reset(self):
        """Reset the internal log dictionary and timer."""
        self.t_0 = None
        self.log_dict = {}

    def flush(self):
        """
        Compute mean of accumulated logs and reset.

        Returns:
            dict: Dictionary containing the mean of logged values.
        """
        ret = {k: np.mean(v) for k, v in self.log_dict.items()}
        self.reset()
        return ret

    def __call__(self, kwargs):
        """
        Call method alias for `log`.

        Args:
            kwargs (dict): Key-value pairs to log.
        """
        self.log(kwargs)



def silent_print(*args, sep=' ', end='\n', file=sys.stdout):
    """
    A drop-in replacement for print() that bypasses W&B console capture.
    Supports sys.stdout and sys.stderr.

    Args:
        *args: Objects to print.
        sep (str, optional): Separator between objects. Defaults to ' '.
        end (str, optional): End character/string. Defaults to '\n'.
        file (file-like, optional): Output stream, either sys.stdout or sys.stderr. Defaults to sys.stdout.

    Raises:
        ValueError: If file is not sys.stdout or sys.stderr.
    """
    # 1. Determine the file descriptor
    # Standard: 1 for stdout, 2 for stderr
    if file is sys.stderr:
        fd = 2
    elif file is sys.stdout:
        fd = 1
    else:
        raise ValueError("Unsupported file object. Only sys.stdout and sys.stderr are supported.")

    # 2. Stringify and join arguments
    message = sep.join(map(str, args)) + end

    # 3. Write directly to the hardware stream
    os.write(fd, message.encode())