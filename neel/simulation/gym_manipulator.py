# !/usr/bin/env python
"""

Created by Indraneel on 11/19/2025

Gym environment for mujoco

"""

import logging
from dataclasses import dataclass
from typing import Any

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
)
from lerobot.processor.converters import identity_transition

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

    pass

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

    control_loop()
    


if __name__ == "__main__":
    main()