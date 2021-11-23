import wandb
import os, sys

def support_unobserve():
    if "--unobserve" in sys.argv:
        sys.argv.remove("--unobserve")
        os.environ["WANDB_MODE"] = "dryrun"

def init(project_name, config, entity=None, tags=None, **kwargs):
    if entity is None:
        entity = os.environ['WANDB_ENTITY']
    wandb.init(project=PROJECT_NAME, entity=entity,
               config=args, tags=args.tags, **kwargs)