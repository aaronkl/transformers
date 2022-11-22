"""
An entry point that loads a serialized function from `PythonBackend` and executes it with the provided hyperparameter.
The md5 hash of the file is first checked before executing the deserialized function.
"""
import os
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Dict
import time
import dill
import hashlib

from syne_tune.config_space import add_to_argparse, from_dict
from syne_tune.report import Reporter
from syne_tune.constants import ST_CHECKPOINT_DIR

from transformers import *
from datasets import *

report = Reporter()


def file_md5(filename: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(f"--{ST_CHECKPOINT_DIR}", type=str)

    root = Path(__file__).parent

    st = time.time()

    with open(root / "trainer.dill", "rb") as file:
        trainer = dill.load(file)

    with open(root / "configspace.json", "r") as file:
        config_space = json.load(file)
        config_space = {k: from_dict(v) if isinstance(v, Dict) else v for k, v in config_space.items()}

    loading_time = time.time() - st
    print('loading time:', loading_time)

    trainer.loading_time = loading_time

    add_to_argparse(parser, config_space)
    args, _ = parser.parse_known_args()

    assert file_md5(root / "trainer.dill") == args.tune_function_hash, \
        "The hash of the Trainer should match the hash obtained when serializing in Syne Tune."

    params = vars(args)
    if os.path.exists(params[ST_CHECKPOINT_DIR]) and len(os.listdir(params[ST_CHECKPOINT_DIR])) > 0:
        checkpoint = params[ST_CHECKPOINT_DIR]
    else:
        checkpoint = None

    trainer.objective = None

    trial = {k: v for k, v in params.items() if k in config_space}
    trial['trial_id'] = int(params[ST_CHECKPOINT_DIR].split('/')[-2])
    trial.pop('tune_function_hash')
    print(trial)
    st = time.time()
    trainer.train(resume_from_checkpoint=checkpoint, trial=trial)
    training_time = time.time() - st

    st = time.time()

    # If there hasn't been any evaluation during the training loop.
#     if getattr(trainer, "objective", None) is None:
#         metrics = trainer.evaluate()
#         trainer.objective = trainer.compute_objective(metrics)
#         evaluation_time = time.time() - st
#         report(objective=trainer.objective, loading_time=loading_time,
#                training_time=training_time, evaluation_time=evaluation_time)




#     if config_space['evaluate_only_at_the_end']:
#         report(objective=trainer.objective, loading_time=loading_time,
#                training_time=training_time)
