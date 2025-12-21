# !/usr/bin/env python
"""
Created by Indraneel on 12/21/2025

Learner process which initialises the policy and replay buffer and updates policy
based on transitions received from the actor server

"""
import logging
import os

from termcolor import colored

from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.configs import parser
from lerobot.utils.utils import (
    init_logging
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR, 
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR
)

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
    cfg.validate()

    if job_name is None:
        job_name = cfg.job_name

    if job_name is None:
        raise ValueError("Job name must be specified either in config or as parameter")
    
    display_pid = False
    if not use_threads(cfg):
        display_pid = True

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"learner_{job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Learner logging initialized, writing to {log_file}")

    # Setup WandB logging if enabled
    if cfg.wandb.enable and cfg.wandb.project:
        from lerobot.rl.wandb_utils import WandBLogger

        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally", "yellow", attrs=["bold"]))
    
    # Handle resume logic
    cfg = handle_resume_logic(cfg)


def handle_resume_logic(cfg: TrainRLServerPipelineConfig) -> TrainRLServerPipelineConfig:
    """
    Handle the resume logic
    """
    out_dir = cfg.output_dir
    checkpoint_dir = os.path.join(out_dir, CHECKPOINTS_DIR, LAST_CHECKPOINT_LINK)
    
    # Case 1
    if not cfg.resume:
        if os.path.exists(checkpoint_dir):
            raise RuntimeError(
                f"Output directory {checkpoint_dir} already exists. Use `resume=true` to resume training."
            )
        return cfg
    
    # Case 2
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"No model checkpoint found in {checkpoint_dir} for resume=True")
    
    # Log that we found a valid checkpoint and are resuming
    logging.info(
        colored(
            "Valid checkpoint found: resume=True detected, resuming previous run",
            color="yellow",
            attrs=["bold"]
        )
    )

    # Load config using Draccus
    checkpoint_cfg_path = os.path.join(checkpoint_dir, PRETRAINED_MODEL_DIR, "train_config.json")
    checkpoint_cfg = TrainRLServerPipelineConfig.from_pretrained(checkpoint_cfg_path)

    # Ensure resume flag is set in returned config
    checkpoint_cfg.resume = True
    return checkpoint_cfg



if __name__ == "__main__":
    train_cli()
    logging.info("[LEARNER] main finished")

