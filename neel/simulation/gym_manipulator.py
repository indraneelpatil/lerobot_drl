# !/usr/bin/env python
"""

Created by Indraneel on 11/19/2025

Gym environment for mujoco

python simulation/gym_manipulator.py --config_path simulation/config/gym_hil_env.json

"""

import logging
from dataclasses import dataclass
from typing import Any
import time
import torch

import gymnasium as gym
from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    InterventionActionProcessorStep,
    Torch2NumpyActionProcessorStep,
    Numpy2TorchActionProcessorStep,
    VanillaObservationProcessorStep,
    create_transition,
    TransitionKey
)
from lerobot.processor.converters import identity_transition
from lerobot.utils.robot_utils import busy_wait
from lerobot.teleoperators import (
    keyboard
)

logging.basicConfig(level=logging.INFO)

@dataclass
class DatasetConfig:
    """ Configuration for dataset creation and management."""

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

def make_robot_env(cfg: HILSerlRobotEnvConfig) -> tuple[gym.Env, Any]:
    """Create robot environment
    
    """
    if cfg.name == "gym_hil":
        assert cfg.robot is None and cfg.teleop is None, "GymHIL environment does not support robot or teleop"
        import gym_hil

        # Extract gripper settings
        use_gripper = cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else True
        gripper_penalty = cfg.processor.gripper.gripper_penalty if cfg.processor.gripper is not None else 0.0

        env = gym.make(
            f"gym_hil/{cfg.task}",
            image_obs=True,
            render_mode="human",
            use_gripper=use_gripper,
            gripper_penalty=gripper_penalty,
        )

        return env, None

    # TODO Real robot environment
    raise NotImplementedError("Real robot environment not implemented yet")


def make_processors(
        env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str ="cpu"
) -> tuple[DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]]:
    """Create action processors"""
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    if cfg.name == "gym_hil":
        action_pipeline_steps = [
            InterventionActionProcessorStep(terminate_on_success=terminate_on_success),
            Torch2NumpyActionProcessorStep()
        ]

        env_pipeline_steps = [
            Numpy2TorchActionProcessorStep(),
            VanillaObservationProcessorStep(),
            AddBatchDimensionProcessorStep(),
            DeviceProcessorStep(device=device)
        ]
    
        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition)

    raise NotImplementedError("Real robot processors not implemented")


def step_env_and_process_transition(
    env: gym.Env,
    transition: EnvTransition,
    action: torch.Tensor,
    env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
    action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
) -> EnvTransition:
    """
    Execute one step with processor pipeline
    """

    # Create action transition
    transition[TransitionKey.ACTION] = action
    transition[TransitionKey.OBSERVATION] = (
        env.get_raw_joint_positions() if hasattr(env, "get_raw_joint_positions") else {}
    )
    processed_action_transition = action_processor(transition)
    processed_action = processed_action_transition[TransitionKey.ACTION]

    obs, reward, terminated, truncated, info = env.step(processed_action)

    reward = reward + processed_action_transition[TransitionKey.REWARD]
    terminated = terminated or processed_action_transition[TransitionKey.DONE]
    truncated = truncated or processed_action_transition[TransitionKey.TRUNCATED]
    complementary_data= processed_action_transition[TransitionKey.COMPLEMENTARY_DATA]
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=processed_action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data
    )
    new_transition = env_processor(new_transition)

    return new_transition



def control_loop(env: gym.Env, 
                 env_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
                 action_processor: DataProcessorPipeline[EnvTransition, EnvTransition],
                 teleop_device: Teleoperator,
                 cfg: GymManipulatorConfig
                 ) -> None:
    """ Main control loop for environment interaction"""
    dt = 1.0 / cfg.env.fps

    print(f"Starting control loop at {cfg.env.fps} FPS")
    print("Controls:")
    print("- Use teleop device for intervention")
    print("- When not intervening, robot will stay still")
    print("- Press Ctrl+C to exit")

    # Reset environment and processors
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

    # TODO: Implement record

    episode_idx = 0
    episode_step = 0
    episode_start_time = time.perf_counter()

    while episode_idx < cfg.dataset.num_episodes_to_record:
        step_start_time = time.perf_counter()

        # Create a neutral action 
        neutral_action = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        if use_gripper:
            neutral_action = torch.cat([neutral_action, torch.tensor([1.0])]) # Gripper stay
        
        # Use the new step function
        transition = step_env_and_process_transition(
            env=env,
            transition=transition,
            action=neutral_action,
            env_processor=env_processor,
            action_processor=action_processor,
        )
        terminated = transition.get(TransitionKey.DONE, False)
        truncated = transition.get(TransitionKey.TRUNCATED, False)

        # TODO Record mode stuff

        episode_step += 1

        # Handle termination
        if terminated or truncated:
            episode_time = time.perf_counter() - episode_start_time
            logging.info(
                f"Episode ended after {episode_step} steps in {episode_time:.1f}s with reward {transition[TransitionKey.REWARD]}"            
            )
            episode_step = 0
            episode_idx += 1

            # TODO Dataset stuff

            # Reset for new episode
            obs, info = env.reset()
            env_processor.reset()
            action_processor.reset()


            transition = create_transition(observation=obs, info=info)
            transition = env_processor(transition)

        # Maintain fps timing
        busy_wait(dt -(time.perf_counter() - step_start_time))

    # TODO Push dataset

@parser.wrap()
def main(cfg: GymManipulatorConfig) -> None:
    """ Main entry """
    env, teleop_device = make_robot_env(cfg.env)
    env_processor, action_processor = make_processors(env, teleop_device, cfg.env, cfg.device)

    print("Environment observation space:", env.observation_space)
    print("Environment Action Space:", env.action_space)
    print("Environment processor:", env_processor)
    print("Action processor:", action_processor)

    # TODO Implement replay mode

    control_loop(env, env_processor, action_processor, teleop_device, cfg)
    


if __name__ == "__main__":
    main()