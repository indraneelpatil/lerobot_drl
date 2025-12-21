# !/usr/bin/env python
"""
Created by Indraneel on 12/21/2025

Learner process which initialises the policy and replay buffer and updates policy
based on transitions received from the actor server

"""
import logging

from lerobot.configs import parser

@parser.wrap()
def train_cli(cfg: TrainRLServerPipelineConfig):
    if not use_threads(cfg):
        import torch.multiprocessing as mp
        mp.set_start_method("spawn")

    # Use the job name from the config
    train(
        cfg,
        job_name=cfg.job_name,
    )
    logging.info("[LEARNER] train_cli finished")

def use_threads(cfg: TrainRLServerPipelineConfig) -> bool:
    return cfg.policy.concurrency.learner == "threads"

def train(cfg: TrainRLServerPipelineConfig, job_name: str | None = None):
    """
    Main training function that initialized and runs the training process
    """
    pass

if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")

