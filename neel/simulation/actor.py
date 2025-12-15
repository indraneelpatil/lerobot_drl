# !/usr/bin/env python
"""
Created by Indraneel on 12/15/2025

Actor process to interact with environment with current policy and collect interactions

"""
import os
import logging
from functools import lru_cache

from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.utils.utils import (
    init_logging
)
from lerobot.rl.process import ProcessSignalHandler

def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.actor == "threads"

@parser.wrap()
def actor_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()
    display_pid = False
    if not use_threads(cfg):
        import torch.multiprocessing as mp

        mp.set_start_method("spawn")
        display_pid = True
    
    # Create logs directory
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"actor_{cfg.job_name}.log")

    #Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Actor logging initialized, writing to {log_file}")

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    learner_client, grpc_channel = learner_service_client(
        host=cfg.policy.actor_learner_config.learner_host.
        port=cfg.policy.actor_learner_config.learner_port,
    )

# Run function only once and cache the result
@lru_cache(max_size=1)
def learner_service_client(
):
    pass


if __name__ == "__main__":
    actor_cli()