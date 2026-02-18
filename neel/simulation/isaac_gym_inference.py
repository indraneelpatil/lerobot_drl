# !/usr/bin/env python
"""
Created by Indraneel on 02/17/2026


Inference process in isaac sim

python -m simulation.isaac_gym_inference \
  --resume=true \
  --config_path=outputs/train/2026-02-15/15-40-38_lerobot_rl_sim_sac/checkpoints/last/pretrained_model/train_config.json

"""

"""Launch Isaac Sim Simulator first."""
import multiprocessing
if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

# isaac sim args
app_launcher_args = {
    "enable_cameras" : True
}

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


"""Rest everything else follows."""

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import TerminationTermCfg, DatasetExportMode
from isaaclab.sim import SimulationCfg, SimulationContext



import logging
import os
import time
from functools import lru_cache
from queue import Empty

import grpc
import torch
from torch import nn
from torch.multiprocessing import Event, Queue

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.policies.factory import make_policy
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.processor import TransitionKey
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.queue import get_last_item_from_queue
from lerobot.robots import so100_follower  # noqa: F401
from lerobot.teleoperators import gamepad, so101_leader  # noqa: F401
from lerobot.teleoperators.utils import TeleopEvents
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import (
    bytes_to_state_dict,
    grpc_channel_options,
    python_object_to_bytes,
    receive_bytes_in_chunks,
    send_bytes_in_chunks,
    transitions_to_bytes,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.transition import (
    Transition,
    move_state_dict_to_device,
    move_transition_to_device,
)
from lerobot.utils.utils import (
    TimerManager,
    get_safe_torch_device,
    init_logging,
)

from .isaac_gym_utils import (
    create_transition,
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

from .actor import (
    establish_learner_connection,
    learner_service_client,
    receive_policy,
    send_transitions,
    send_interactions,
    use_threads,
    log_policy_frequency_issue,
    update_policy_parameters,
    push_transitions_to_transport_queue,
    get_frequency_stats
)

from .learner import (
    handle_resume_logic
)


@parser.wrap()
def inference_cli(cfg: TrainRLServerPipelineConfig):
    cfg.validate()

    display_pid = False

    # Create logs directory to ensure it exists
    log_dir = os.path.join(cfg.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"inference_{cfg.job_name}.log")

    # Initialize logging with explicit log file
    init_logging(log_file=log_file, display_pid=display_pid)
    logging.info(f"Inference logging initialized, writing to {log_file}")

    # Handle resume logic
    cfg = handle_resume_logic(cfg)

    is_threaded = use_threads(cfg)
    shutdown_event = ProcessSignalHandler(is_threaded, display_pid=display_pid).shutdown_event

    act_with_policy(
        cfg=cfg,
        shutdown_event=shutdown_event
    )

    logging.info("[INFERENCE] Policy process joined")


def act_with_policy(
    cfg: TrainRLServerPipelineConfig,
    shutdown_event: any,  # Event,
):
    """
    Executes policy interaction within the environment

    """
    logging.info("make_env online")

    online_env, teleop_device = make_robot_env(cfg.env, cfg.policy.device)
    env_processor, action_processor = make_processors(online_env, teleop_device, cfg.env, cfg.policy.device)

    set_seed(cfg.seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("make_policy")

    policy: SACPolicy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy = policy.eval()
    assert isinstance(policy, nn.Module)
    
    obs, info = online_env.reset()
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info)
    transition = env_processor(transition)

    # NOTE: For the moment we will solely handle the case of a single environment
    sum_reward_episode = 0
    episode_total_steps = 0

    policy_timer = TimerManager("Policy inference", log=False)

    for interaction_step in range(cfg.policy.online_steps):
        start_time = time.perf_counter()
        if shutdown_event.is_set():
            logging.info("[ACTOR] Shutting down act_with_policy")
            return
        
        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }

        # Time policy inference and check if it meets FPS requirement
        with policy_timer:
            # Extract observation from transition for policy
            action = policy.select_action(batch=observation)
        policy_fps = policy_timer.fps_last

        log_policy_frequency_issue(policy_fps=policy_fps, cfg=cfg, interaction_step=interaction_step)

        # Use the new step function
        new_transition = step_env_and_process_transition(
            env=online_env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        reward = new_transition[TransitionKey.REWARD]
        done = new_transition.get(TransitionKey.DONE, False)
        truncated = new_transition.get(TransitionKey.TRUNCATED, False)

        sum_reward_episode += float(reward)
        episode_total_steps += 1

        # Update transition for next iteration
        transition = new_transition
        
        if done or truncated:
            logging.info(f"[INFERENCE] Global step {interaction_step}: Episode reward: {sum_reward_episode}")

            stats = get_frequency_stats(policy_timer)
            policy_timer.reset()

            # Reset intervention counters and environment
            sum_reward_episode = 0.0
            episode_total_steps = 0

            # Reset environment and processors
            obs, info = online_env.reset()
            env_processor.reset()
            action_processor.reset()

            # Process initial observation
            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        if cfg.env.fps is not None:
            dt_time = time.perf_counter() - start_time
            busy_wait(1 / cfg.env.fps - dt_time)

    # Close the simulator
    online_env.close()
    simulation_app.close()

if __name__=="__main__":
    inference_cli()