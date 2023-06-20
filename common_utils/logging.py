import wandb
import os, sys
import numpy as np
import time


def support_unobserve():
    if "--unobserve" in sys.argv:
        sys.argv.remove("--unobserve")
        os.environ["WANDB_MODE"] = "dryrun"


def init(config, project_name=None, entity=None, tags=[], notes=None, **kwargs):
    if entity is None:
        assert (
            "WANDB_ENTITY" in os.environ
        ), "Please either pass in \"entity\" to logging.init or set environment variable 'WANDB_ENTITY' to your wandb entity name."
    if project_name is None:
        assert (
            "WANDB_PROJECT" in os.environ
        ), "Please either pass in \"project_name\" to logging.init or set environment variable 'WANDB_PROJECT' to your wandb project name."
    tags.append(os.path.basename(sys.argv[0]))
    if "_MY_JOB_ID" in os.environ:
        x = f"(jobid:{os.environ['_MY_JOB_ID']})"
        notes = x if notes is None else notes + " " + x
    wandb.init(
        project=project_name,
        entity=entity,
        config=config,
        tags=tags,
        notes=notes,
        **kwargs,
    )


class LoggingHandler:
    def __init__(self):
        self.log_count = 0
        self.reset()

    def log(self, kwargs):
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
        self.t_0 = None
        self.log_dict = {}

    def flush(self):
        ret = {k: np.mean(v) for k, v in self.log_dict.items()}
        self.reset()
        return ret

    def __call__(self, kwargs):
        self.log(kwargs)
