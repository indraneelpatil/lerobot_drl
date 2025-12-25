# !/usr/bin/env python
"""
Created by Indraneel on 12/21/2025

Learner process which initialises the policy and replay buffer and updates policy
based on transitions received from the actor server

"""
import logging
import os
import time

import grpc
from termcolor import colored
import torch
from torch import nn
from torch.multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor
from torch.optim.optimizer import Optimizer

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.factory import make_dataset
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.configs import parser
from lerobot.utils.utils import (
    init_logging,
    format_big_number
)
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR, 
    LAST_CHECKPOINT_LINK,
    PRETRAINED_MODEL_DIR,
    TRAINING_STATE_DIR
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.transition import move_state_dict_to_device
from lerobot.rl.buffer import ReplayBuffer
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.transport.utils import (
    MAX_MESSAGE_SIZE,
    state_to_bytes
)
from lerobot.transport import services_pb2_grpc
from lerobot.utils.utils import (
    get_safe_torch_device
)
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.policies.factory import make_policy

from .learner_service import MAX_WORKERS, SHUTDOWN_TIMEOUT, LearnerService

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

    set_seed(seed=cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event
    )

def start_learner_threads(
        cfg: TrainRLServerPipelineConfig,
        wandb_logger: WandBLogger | None,
        shutdown_event: any, # Event
) -> None:
    """
    Start the learner threads for training
    """
    # Create multiprocessing queues
    transition_queue = Queue()
    interaction_message_queue = Queue()
    parameters_queue = Queue()

    concurrency_entity = None

    if use_threads(cfg):
        from threading import Thread

        concurrency_entity = Thread
    else:
        from torch.multiprocessing import Process

        concurrency_entity = Process

    communication_process = concurrency_entity(
        target=start_learner,
        args=(
              parameters_queue,
              transition_queue,
              interaction_message_queue,
              shutdown_event,
              cfg,
            ),
            daemon=True
    )
    communication_process.start()

    add_actor_information_and_train(
    )
    logging.info("[LEARNER] Training process stopped")

    logging.info("[LEARNER] Closing queues")
    transition_queue.close()
    interaction_message_queue.close()
    parameters_queue.close()

    communication_process.join()
    logging.info("[LEARNER] Communication process joined")
    
    logging.info("[LEARNER] join queues")
    transition_queue.cancel_join_thread()
    interaction_message_queue.cancel_join_thread()
    parameters_queue.cancel_join_thread()

    logging.info("[LEARNER] queues closed")

def start_learner(
    parameters_queue: Queue,
    transition_queue: Queue,
    interaction_message_queue: Queue,
    shutdown_event: any, # Event
    cfg: TrainRLServerPipelineConfig,
):
    """
    Start the learner server for training
    """
    if not use_threads(cfg):
        # Create a process-specific log file
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_process_{os.getpid()}.log")

        # Initialize logging with explicit log file
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Learner server process logging initialized")

        # Setup process handlers to handle shutdown signal
        # But use shutdown event from the main process
        _ = ProcessSignalHandler(False, display_pid=True)
    
    service = LearnerService(
        shutdown_event=shutdown_event,
        parameters_queue=parameters_queue,
        seconds_between_pushes=cfg.policy.actor_learner_config.policy_parameters_push_frequency,
        transition_queue=transition_queue,
        interaction_message_queue=interaction_message_queue,
        queue_get_timeout=cfg.policy.actor_learner_config.queue_get_timeout,
    )

    server = grpc.server(
        ThreadPoolExecutor(max_workers=MAX_WORKERS),
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_SIZE),
            ("grpc.max_send_message_length", MAX_MESSAGE_SIZE)
        ]
    )

    services_pb2_grpc.add_LearnerServiceServicer_to_server(
        service,
        server
    )

    host = cfg.policy.actor_learner_config.learner_host
    port = cfg.policy.actor_learner_config.learner_port

    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logging.info("[LEARNER] gRPC server started")

    shutdown_event.wait()
    logging.info("[LEARNER] Stopping gRPC server...")
    server.stop(SHUTDOWN_TIMEOUT)


def add_actor_information_and_train(
    cfg: TrainRLServerPipelineConfig,
    wandb_logger: WandBLogger | None,
    shutdown_event: any, # Event
    transition_queue: Queue,
    interaction_message_queue: Queue,
    parameters_queue: Queue 
):
    """
    Fill replay buffer from actor, sample batches from buffers and perform
    critic updates, also update actor and other optimizers, log training
    statistics
    """
    # Extract all configuration variables at the beginning, it improves the 
    # speed performance of 7%
    device = get_safe_torch_device(try_device=cfg.policy.device, log=True)
    storage_device = get_safe_torch_device(try_device=cfg.policy.storage_device)
    clip_grad_norm_value = cfg.policy.grad_clip_norm
    online_step_before_learning = cfg.policy.online_step_before_learning
    utd_ratio = cfg.policy.utd_ratio
    fps = cfg.env.fps
    log_freq = cfg.log_freq
    save_freq = cfg.save_freq
    policy_update_freq = cfg.policy.policy_update_freq
    policy_parameters_push_frequency = cfg.policy.actor_learner_config.policy_parameters_push_frequency
    saving_checkpoint = cfg.save_checkpoint
    online_steps = cfg.policy.online_steps
    async_prefetch = cfg.policy.async_prefetch

    # Initialize logging for multiprocessing
    if not use_threads(cfg):
        log_dir = os.path.join(cfg.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"learner_train_process_{os.getpid()}.log")
        init_logging(log_file=log_file, display_pid=True)
        logging.info("Initialized logging for actor information and training process")

    logging.info("Initializing policy")

    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env
    )

    assert isinstance(policy, nn.Module)

    policy.train()

    push_actor_policy_to_queue(parameters_queue=parameters_queue, policy=policy)

    last_time_policy_pushed = time.time()

    optimizers, lr_scheduler = make_optimizers_and_scheduler(cfg=cfg, policy=policy)

    # If we are resuming, we need to load the training state
    resume_optimization_step, resume_interaction_step = load_training_state(cfg=cfg, optimizers=optimizers)

    log_training_info(cfg=cfg, policy=policy)

    replay_buffer = initialize_replay_buffer(cfg, device, storage_device)
    batch_size = cfg.batch_size
    offline_replay_buffer = None

    if cfg.dataset is not None:
        offline_replay_buffer = initialize_offline_replay_buffer(
            cfg=cfg,
            device=device,
            storage_device=storage_device
        )
        batch_size: int = batch_size // 2 # We will sample from both the replay buffer

    logging.info("Starting learner thread")

def push_actor_policy_to_queue(parameters_queue: Queue, policy: nn.Module):
    logging.debug("[LEARNER] Pushing actor policy to the queue")

    # Create a dictionary to hold all the state dicts
    state_dicts = {"policy": move_state_dict_to_device(policy.actor.state_dict(), device="cpu")}

    # Add discrete critic if it exists
    if hasattr(policy, "discrete_critic") and policy.discrete_critic is not None:
        state_dicts["discrete_critic"] = move_state_dict_to_device(
            policy.discrete_critic.state_dict(), device="cpu"
        )
        logging.debug("[LEARNER] Including discrete critic in state dict push")

    state_bytes = state_to_bytes(state_dicts)
    parameters_queue.put(state_bytes)


def make_optimizers_and_scheduler(cfg: TrainRLServerPipelineConfig, policy: nn.Module):
    """
    Initialize Adam optimizers for actor, critic and temperature
    
    """
    optimizer_actor = torch.optim.Adam(
        params=[
            p
            for n, p in policy.actor.named_parameters()
            if not policy.config.shared_encoder or not n.startswith("encoder")
        ],
        lr=cfg.policy.actor_lr
    )
    optimizer_critic = torch.optim.Adam(params=policy.critic_ensemble.parameters(), lr=cfg.policy.critic_lr)

    if cfg.policy.num_discrete_actions is not None:
        optimizer_discrete_critic = torch.optim.Adam(
            params=policy.discrete_critic.parameters(), lr=cfg.policy.critic_lr
        )
    optimizer_temperature = torch.optim.Adam(params=[policy.log_alpha], lr=cfg.policy.critic_lr)
    lr_scheduler = None
    optimizers = {
        "actor": optimizer_actor,
        "critic": optimizer_critic,
        "temperature": optimizer_temperature,
    }
    if cfg.policy.num_discrete_actions is not None:
        optimizers["discrete_critic"] = optimizer_discrete_critic
    return optimizers, lr_scheduler

def load_training_state(
        cfg: TrainRLServerPipelineConfig,
        optimizers: Optimizer | dict[str, Optimizer]
):
    """
    Load the training state (optimizer, step count) from a checkpoint
    """
    if not cfg.resume:
        return None, None
    
    raise RuntimeError("Resume Not implemented")
    

def log_training_info(cfg: TrainRLServerPipelineConfig, policy: nn.Module)-> None:
    """
    Log information about the training process
    """
    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.policy.online_steps=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")


def initialize_replay_buffer(
        cfg: TrainRLServerPipelineConfig,
        device: str,
        storage_device: str,
) -> ReplayBuffer:
    """
    Initialize a replay buffer, either empty or from a dataset if resuming.
    """
    if not cfg.resume:
        return ReplayBuffer(
            capacity=cfg.policy.online_buffer_capacity,
            device=device,
            state_keys=cfg.policy.input_features.keys(),
            storage_device=storage_device,
            optimize_memory=True

        )
    
    raise RuntimeError("Resuming not implemented")


def initialize_offline_replay_buffer(
    cfg: TrainRLServerPipelineConfig,
    device: str,
    storage_device:str,
) -> ReplayBuffer:
    """
    Initialize offline replay buffer from a dataset
    """
    if not cfg.resume:
        logging.info("make dataset offline buffer")
        offline_dataset = make_dataset(cfg)
    else:
        logging.info("load offline dataset")
        dataset_offline_path = os.path.join(cfg.output_dir, "dataset_offline")
        offline_dataset = LeRobotDataset(
            repo_id = cfg.dataset.repo_id,
            root=dataset_offline_path
        )
    
    logging.info("Convert to a offline replay buffer")
    offline_replay_buffer = ReplayBuffer.from_lerobot_dataset(
        offline_dataset,
        device=device,
        state_keys=cfg.policy.input_features.keys(),
        storage_device=storage_device,
        optimize_memory=True,
        capacity=cfg.policy.offline_buffer_capacity,
    )
    return offline_replay_buffer


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

