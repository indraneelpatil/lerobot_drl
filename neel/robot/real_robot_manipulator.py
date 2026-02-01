#!/usr/bin/env python
"""
Real robot data collection script for SO101 follower + leader.

Usage:
    python -m neel.robot.real_robot_manipulator --config_path neel/simulation/config/real_robot_env_record.json
"""

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
    so_leader,  # noqa: F401
)
from lerobot.cameras import opencv  # noqa: F401

import lerobot.teleoperators.so_leader.so_leader
import logging
from dataclasses import dataclass
from typing import Any
import time
import torch
import numpy as np

import gymnasium as gym
from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.processor import (
    DataProcessorPipeline,
    EnvTransition,
    create_transition,
    TransitionKey,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.teleoperators.utils import TeleopEvents

# Import from isaac_gym_utils (which now has real_robot support)
import sys
sys.path.append('/Users/sisaha/Documents/Personal dev/physicalaihack/lerobot_drl')
from neel.simulation.isaac_gym_utils import (
    make_processors,
    make_robot_env,
    step_env_and_process_transition,
)

logging.basicConfig(level=logging.INFO)

hc_joint_names = ["shoulder_pan",
                  "shoulder_lift",
                  "elbow_flex",
                  "wrist_flex",
                  "wrist_roll",
                  "gripper"]

@dataclass
class DatasetConfig:
    """Configuration for dataset creation and management."""

    repo_id: str
    task: str
    root: str | None = None
    num_episodes_to_record: int = 5
    replay_episode: int | None = None
    push_to_hub: bool = False


@dataclass
class GymManipulatorConfig:
    env: HILSerlRobotEnvConfig
    dataset: DatasetConfig
    mode: str | None = None
    device: str = "cpu"


leader_torque_disabled = False

def mirror_follower_if_in_auto(teleop_device: Teleoperator | None, transition: EnvTransition, env: gym.Env):
    global leader_torque_disabled
    if not isinstance(teleop_device, lerobot.teleoperators.so_leader.so_leader.SOLeader):
        return
    # Mirror follower arm to leader arm when teleop is off
    if teleop_device is not None and teleop_device.is_connected:
        info = transition.get(TransitionKey.INFO, {})
        is_intervention = info.get(TeleopEvents.IS_INTERVENTION, False)
        
        # If teleop is off (no intervention), mirror follower to leader
        if not is_intervention:
            try:
                # Get follower arm joint positions from environment
                if hasattr(env, "get_raw_joint_positions"):
                    follower_joint_positions = env.get_raw_joint_positions()
                    
                    # Convert from {"shoulder_pan.pos": value, ...} to {"shoulder_pan": value, ...}
                    # for the leader arm bus.sync_write format
                    leader_goal_positions = {}
                    for joint_name in hc_joint_names:
                        joint_key = f"{joint_name}.pos"
                        if joint_key in follower_joint_positions:
                            leader_goal_positions[joint_name] = follower_joint_positions[joint_key]
                    
                    # Send positions to leader arm
                    if leader_goal_positions and hasattr(teleop_device, "bus"):
                        teleop_device.bus.sync_write("Goal_Position", leader_goal_positions)
                        leader_torque_disabled = False
            except Exception as e:
                logging.debug(f"Error mirroring follower to leader: {e}")
        else:
            # Allow leader to move freely when in intervention mode
            if not leader_torque_disabled:
                leader_torque_disabled = True
                teleop_device.bus.disable_torque()

def control_loop(
    env: gym.Env,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    cfg: GymManipulatorConfig,
    teleop_device: Teleoperator | None = None,
):
    """Main control loop for real robot interaction."""

    dt = 1.0 / cfg.env.fps

    print(f"Starting real robot control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use leader arm for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press 's' for success, 'f' for failure, 'r' to rerecord")
    print("- Press Ctrl+C to exit")

    # Reset environment
    obs, info = env.reset()
    complimentary_data = (
        {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
    )
    env_processor.reset()
    action_processor.reset()

    # Process initial observation
    transition = create_transition(observation=obs, info=info, complementary_data=complimentary_data)
    transition = env_processor(data=transition)

    # Check gripper
    use_gripper = cfg.env.processor.gripper.use_gripper if cfg.env.processor.gripper is not None else True

    dataset = None
    if cfg.mode == "record":
        features = {
            ACTION: {"dtype": "float32", "shape": (4,), "names": None},
            REWARD: {"dtype": "float32", "shape": (1,), "names": None},
            DONE: {"dtype": "bool", "shape": (1,), "names": None},
        }
        if use_gripper:
            features["complementary_info.discrete_penalty"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["discrete_penalty"],
            }

        for key, value in transition[TransitionKey.OBSERVATION].items():
            if key == OBS_STATE:
                features[key] = {
                    "dtype": "float32",
                    "shape": value.squeeze(0).shape,
                    "names": None
                }
            if "image" in key:
                features[key] = {
                    "dtype": "video",
                    "shape": value.squeeze(0).shape,
                    "names": ["channels", "height", "width"]
                }

        # Create dataset
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.env.fps,
            root=cfg.dataset.root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            features=features
        )

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    print("#" * 50)
    print("Starting real robot control loop")
    print("#" * 50)

    # Main control loop
    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        # Create a neutral action (robot stays still unless intervened)
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])])  # Gripper stay

        # Execute step with processor pipeline
        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        terminated = terminated.squeeze(0).cpu() if isinstance(terminated, torch.Tensor) else terminated
        truncated = truncated.squeeze(0).cpu() if isinstance(truncated, torch.Tensor) else truncated
        
        mirror_follower_if_in_auto(teleop_device, transition, env)

        if cfg.mode == "record":
            observations = {
                k: v.squeeze(0).cpu() if isinstance(v, torch.Tensor) else v
                for k, v in transition[TransitionKey.OBSERVATION].items()
            }

            action = transition[TransitionKey.ACTION].squeeze(0).cpu() if isinstance(transition[TransitionKey.ACTION], torch.Tensor) else transition[TransitionKey.ACTION]
            reward = transition[TransitionKey.REWARD].squeeze(0).cpu() if isinstance(transition[TransitionKey.REWARD], torch.Tensor) else np.array([transition[TransitionKey.REWARD]], dtype=np.float32)
            done = torch.tensor([terminated or truncated], dtype=torch.bool)

            # Add to dataset
            dataset_frame = {
                **observations,
                ACTION: action,
                REWARD: reward,
                DONE: done,
                "task": cfg.dataset.task,
            }

            # Add discrete penalty if using gripper
            if "discrete_penalty" not in transition[TransitionKey.COMPLEMENTARY_DATA]:
                # TODO: Fix this please
                transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"] = np.array([0.0], dtype=np.float32)

            if use_gripper and "discrete_penalty" in transition[TransitionKey.COMPLEMENTARY_DATA]:
                discrete_penalty = transition[TransitionKey.COMPLEMENTARY_DATA]["discrete_penalty"]
                discrete_penalty = discrete_penalty.squeeze(0).cpu() if isinstance(discrete_penalty, torch.Tensor) else discrete_penalty
                dataset_frame["complementary_info.discrete_penalty"] = discrete_penalty

            dataset.add_frame(dataset_frame)

        episode_step += 1

        # Handle episode termination
        if terminated or truncated:
            episode_duration = time.perf_counter() - episode_start_time
            logging.info(f"Episode {episode_idx + 1} finished in {episode_step} steps ({episode_duration:.2f}s)")

            if cfg.mode == "record":
                dataset.save_episode()
                logging.info(f"Saved episode {episode_idx + 1} to dataset")

            episode_idx += 1
            episode_step = 0

            if episode_idx < cfg.dataset.num_episodes_to_record:
                # Reset for next episode
                logging.info(f"Starting episode {episode_idx + 1}/{cfg.dataset.num_episodes_to_record}")
                obs, info = env.reset()
                complimentary_data = (
                    {"raw_joint_positions": info.pop("raw_joint_positions")} if "raw_joint_positions" in info else {}
                )
                env_processor.reset()
                action_processor.reset()

                transition = create_transition(observation=obs, info=info, complementary_data=complimentary_data)
                transition = env_processor(data=transition)
                episode_start_time = time.perf_counter()

        # Maintain FPS
        step_duration = time.perf_counter() - step_start_time
        busy_wait(dt - step_duration)

    # Push to HuggingFace Hub if configured
    if cfg.mode == "record" and cfg.dataset.push_to_hub:
        logging.info("Pushing dataset to HuggingFace Hub...")
        dataset.push_to_hub()
        logging.info("Dataset pushed successfully")

    logging.info("Control loop completed")


@parser.wrap()
def main(cfg: GymManipulatorConfig):
    """Main entry point for real robot data collection."""

    logging.info("=" * 50)
    logging.info("Real Robot Data Collection")
    logging.info("=" * 50)

    # Validate configuration
    if cfg.env.name != "real_robot":
        raise ValueError(f"This script is for real_robot environment, got: {cfg.env.name}")

    # Create environment and teleop
    logging.info("Creating real robot environment...")
    env, teleop_device = make_robot_env(cfg.env, device=cfg.device)

    # Create processors
    logging.info("Creating processor pipelines...")
    env_processor, action_processor = make_processors(
        env, teleop_device, cfg.env, cfg.device
    )

    try:
        # Run control loop
        control_loop(env, env_processor, action_processor, cfg, teleop_device)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        # Cleanup
        logging.info("Cleaning up...")
        env.close()
        if teleop_device is not None and hasattr(teleop_device, "disconnect"):
            teleop_device.disconnect()

    logging.info("Finished!")


if __name__ == "__main__":
    main()
