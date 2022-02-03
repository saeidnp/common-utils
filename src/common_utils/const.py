import os

TMPDIR = "/tmp" if "EXP_TMPDIR" not in os.environ else os.environ["EXP_TMPDIR"]