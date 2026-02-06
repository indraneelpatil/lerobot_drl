# !/usr/bin/env python
"""
Created by Indraneel on 01/27/2025
"""

import logging
from dataclasses import dataclass
from typing import Any
import time
import torch
import numpy as np
from pprint import pprint

# Optional imports for Isaac Gym/Sim environments
try:
    import leisaac
    LEISAAC_AVAILABLE = True
except ImportError:
    LEISAAC_AVAILABLE = False

try:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_tasks.utils import parse_env_cfg
    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False

import gymnasium as gym
from lerobot.configs import parser
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    AddTeleopActionAsComplimentaryDataStep,
    AddTeleopEventsAsInfoStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    EnvTransition,
    InterventionActionProcessorStep,
    TeleopConvertJointToDeltaStep,
    Torch2NumpyActionProcessorStep,
    Numpy2TorchActionProcessorStep,
    VanillaObservationProcessorStep,
    create_transition,
    TransitionKey,
    MapTensorToDeltaActionDictStep,
    MapDeltaActionToRobotActionStep,
    RobotActionToPolicyActionProcessorStep,
    GripperPenaltyProcessorStep,
    Degrees2RadiansActionProcessorStep,
    Radians2DegreesObservationProcessor,
    AddProcessorObservationsToState,
    ImageCropResizeProcessorStep
)
from lerobot.processor.pipeline import ProcessorStep
from lerobot.robots.so_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    ForwardKinematicsJointsToEEObservation,
    GripperVelocityToJoint,
    InverseKinematicsRLStep,
)
from lerobot.processor.converters import identity_transition
from lerobot.utils.robot_utils import busy_wait
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    keyboard,  # noqa: F401
    make_teleoperator_from_config,
)
from lerobot.robots import (
    make_robot_from_config,
    # so100_follower,  # noqa: F401
)
from lerobot.utils.constants import ACTION, DONE, OBS_IMAGES, OBS_STATE, REWARD
from lerobot.datasets.lerobot_dataset import LeRobotDataset

hc_joint_names = ["shoulder_pan",
                  "shoulder_lift",
                  "elbow_flex",
                  "wrist_flex",
                  "wrist_roll",
                  "gripper"]



def make_robot_env(cfg: HILSerlRobotEnvConfig, device: str) -> tuple[gym.Env, Any]:
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
            random_block_position=True,
        )

        return env, None
    
    if cfg.name == "gym_isaac_sim_hil" and cfg.task !=None :
        if not ISAACLAB_AVAILABLE:
            raise ImportError(
                "Isaac Lab is not installed. This environment requires isaaclab. "
                "Install it to use gym_isaac_sim_hil environments."
            )
        env_cfg = parse_env_cfg(cfg.task, device, num_envs=1)
        env_cfg.use_teleop_device(cfg.processor.control_mode)
        env_cfg.seed = int(time.time())

        # Modify configuration
        # if hasattr(env_cfg.terminations, "time_out"):
        #     env_cfg.terminations.time_out = None
        # if hasattr(env_cfg.terminations, "success"):
        #     env_cfg.terminations.success = None
        
        # TODO recording
        env_cfg.recorders = None
        
        # Create environment
        env: ManagerBasedRLEnv = gym.make(cfg.task, cfg=env_cfg)

        assert cfg.teleop is not None, "Teleop config must be provided for gym isaac environment"
        teleop_device = make_teleoperator_from_config(cfg.teleop)
        teleop_device.connect()

        # Wrap the environment
        env_wrapped = IsaacSimEnvWrapper(env)

        return env_wrapped, teleop_device

    if cfg.name == "real_robot":
        # Initialize follower arm
        assert cfg.robot is not None, "Robot config must be provided for real robot environment"
        robot = make_robot_from_config(cfg.robot)
        robot.connect()
        logging.info(f"[REAL ROBOT] Connected to follower robot on port {cfg.robot.port}")

        # Initialize leader arm (for interventions)
        teleop_device = None
        if cfg.teleop is not None:
            teleop_device = make_teleoperator_from_config(cfg.teleop)
            teleop_device.connect()
            # Log connection - keyboard teleop doesn't have a port
            if hasattr(cfg.teleop, 'port'):
                logging.info(f"[REAL ROBOT] Connected to leader arm on port {cfg.teleop.port}")
            else:
                logging.info(f"[REAL ROBOT] Connected to teleop device: {cfg.teleop.type}")

        # Extract configuration
        display_cameras = cfg.processor.observation.display_cameras if cfg.processor.observation is not None else False
        fixed_reset_joint_positions = cfg.processor.reset.fixed_reset_joint_positions if cfg.processor.reset is not None else None
        reset_time_s = cfg.processor.reset.reset_time_s if cfg.processor.reset is not None else 2.0
        control_time_s = cfg.processor.reset.control_time_s if cfg.processor.reset is not None else 20.0

        # Create environment wrapper
        env = RobotEnv(
            robot=robot,
            teleop_device=teleop_device,
            display_cameras=display_cameras,
            fixed_reset_joint_positions=fixed_reset_joint_positions,
            reset_time_s=reset_time_s,
            control_time_s=control_time_s,
        )

        return env, teleop_device

    raise NotImplementedError(f"Environment {cfg.name} not implemented yet")



class IsaacSimEnvWrapper(gym.Wrapper):
    def __init__(
        self,
        env,):
        """Init the wrapper"""
        super().__init__(env)
        self._raw_joint_positions = None

    def convert_joint_angle_tensor_to_dict(self, joint_pos):
        joint_pos = joint_pos[0]

        assert len(hc_joint_names) == joint_pos.shape[0], (
            f"Expected {len(hc_joint_names)} joints, got {joint_pos.shape[0]}"
        )

        return {
            f"{name}.pos": joint_pos[i].item()
            for i, name in enumerate(hc_joint_names)
        }
    
    def _get_observation(self, raw_obs):
        # Update raw joint positions
        policy_obs = raw_obs["policy"]
        joint_pos = policy_obs["joint_pos"]  # shape: (B, N)

        joint_angle_dict = self.convert_joint_angle_tensor_to_dict(joint_pos)
        self._raw_joint_positions = joint_angle_dict
        joint_positions = np.array([joint_angle_dict[f"{name}.pos"] for name in hc_joint_names])

        # Process images
        camera_keys = ["front", "wrist"]
        images = {key: policy_obs[key] for key in camera_keys}

        return {"agent_pos": joint_positions, "pixels": images, **joint_angle_dict}

    def step(self, action):
        # Step the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        lerobot_obs = self._get_observation(obs)

        return lerobot_obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment."""
        # Update raw joint positions

        obs, info = self.env.reset(**kwargs)
        lerobot_obs = self._get_observation(obs)

        return lerobot_obs, info
    
    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions in degrees."""

        raw_joint_positions_deg = {}
        for k,v in self._raw_joint_positions.items():
            raw_joint_positions_deg[k] = np.rad2deg(v)
        return raw_joint_positions_deg


class RobotEnv(gym.Env):
    """Gymnasium wrapper for real SO101 follower robot."""

    def __init__(
        self,
        robot,
        teleop_device: Teleoperator | None = None,
        display_cameras: bool = False,
        fixed_reset_joint_positions: list[float] | None = None,
        reset_time_s: float = 2.0,
        control_time_s: float = 20.0,
    ):
        """Initialize the real robot environment.

        Args:
            robot: SO100Follower instance
            teleop_device: Teleoperator device (e.g., SO101 leader arm)
            display_cameras: Whether to display camera feeds
            fixed_reset_joint_positions: Joint positions for reset (in degrees)
            reset_time_s: Time to wait during reset
            control_time_s: Maximum episode duration
        """
        super().__init__()
        self.robot = robot
        self.teleop_device = teleop_device
        self.display_cameras = display_cameras
        self.fixed_reset_joint_positions = fixed_reset_joint_positions
        self.reset_time_s = reset_time_s
        self.control_time_s = control_time_s

        # Episode tracking
        self.current_step = 0
        self.episode_start_time = None
        self._raw_joint_positions = None

        # Setup spaces
        self._setup_spaces()

        logging.info("[RobotEnv] Initialized real robot environment")

    def _setup_spaces(self):
        """Define observation and action spaces."""
        # Action space: 3D end-effector velocities (x, y, z)
        # Actions will be processed through IK pipeline to convert to joint positions
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation space will be defined dynamically based on available cameras
        # For now, we define a simple dict space
        num_joints = len(hc_joint_names)

        # Build observation space dict
        obs_spaces = {
            "agent_pos": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(num_joints,),
                dtype=np.float32
            )
        }

        # Add camera spaces if robot has cameras
        if hasattr(self.robot, "cameras") and self.robot.cameras is not None:
            for camera_name in self.robot.cameras.keys():
                # Assume 480x640 RGB images
                obs_spaces[f"pixels.{camera_name}"] = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(480, 640, 3),
                    dtype=np.uint8
                )

        # Add individual joint position spaces (for compatibility)
        for joint_name in hc_joint_names:
            obs_spaces[f"{joint_name}.pos"] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1,),
                dtype=np.float32
            )

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _get_observation(self):
        """Get current observation from robot."""
        # Read joint positions from robot (in degrees)
        # Use sync_read to get current positions as a dict
        position_dict = self.robot.bus.sync_read("Present_Position")

        # Extract positions in order of joint names
        joint_positions_deg = [position_dict[name] for name in hc_joint_names]

        # Convert to dictionary format
        joint_angle_dict = {}
        for i, name in enumerate(hc_joint_names):
            joint_angle_dict[f"{name}.pos"] = joint_positions_deg[i]

        # Store raw joint positions
        self._raw_joint_positions = joint_angle_dict

        # Create observation dict
        obs = {
            "agent_pos": np.array(list(joint_positions_deg), dtype=np.float32),
            **{k: float(v) for k, v in joint_angle_dict.items()}
        }

        # Add camera images if available
        if hasattr(self.robot, "cameras") and self.robot.cameras is not None:
            images = {}
            for camera_name, camera in self.robot.cameras.items():
                # Read camera image
                image = camera.read()
                if image is not None:
                    images[camera_name] = image
                    # obs[f"pixels.{camera_name}"] = image

                    # Display if requested
                    if self.display_cameras:
                        import cv2
                        cv2.imshow(f"Camera: {camera_name}", image)
                        cv2.waitKey(1)

            # Add 'pixels' key for compatibility
            if images:
                obs["pixels"] = images

        return obs

    def reset(self, seed=None, options=None):
        """Reset the robot to initial position."""
        super().reset(seed=seed)

        logging.info("[RobotEnv] Resetting environment...")

        # Move robot to reset position if specified
        if self.fixed_reset_joint_positions is not None:
            logging.info(f"[RobotEnv] Moving to reset position: {self.fixed_reset_joint_positions}")
            # Create action dict mapping joint names to positions
            action_dict = {
                name: self.fixed_reset_joint_positions[i]
                for i, name in enumerate(hc_joint_names)
            }
            # Use sync_write to send goal positions
            self.robot.bus.sync_write("Goal_Position", action_dict)

            # Wait for robot to reach position
            time.sleep(self.reset_time_s)

        # Reset episode tracking
        self.current_step = 0
        self.episode_start_time = time.time()

        # Get initial observation
        obs = self._get_observation()
        info = {"episode_step": self.current_step}

        logging.info("[RobotEnv] Reset complete")
        return obs, info

    def step(self, action):
        """Execute one step with the given action.

        Args:
            action: Joint positions in degrees (after IK processing)
                   Shape: (num_joints,)

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute action on robot
        # Note: action comes from the action processor pipeline, already converted to joint positions

        if action is not None and len(action) > 0:
            # Create action dict mapping joint names to positions
            import numpy as np
            if not isinstance(action, np.ndarray):
                action = np.array(action)
            action_dict = {
                name: action[i]
                for i, name in enumerate(hc_joint_names)
            }
            # Use sync_write to send goal positions
            self.robot.bus.sync_write("Goal_Position", action_dict)

        # Get new observation
        obs = self._get_observation()



        # Calculate reward (will be overridden by reward classifier or manual annotation)
        reward = 0.0

        # Check termination conditions
        terminated = False
        truncated = False

        # Check time limit
        elapsed_time = time.time() - self.episode_start_time
        if elapsed_time >= self.control_time_s:
            truncated = True
            logging.info(f"[RobotEnv] Episode truncated due to time limit ({self.control_time_s}s)")

        # Update step counter
        self.current_step += 1

        info = {
            "episode_step": self.current_step,
            "elapsed_time": elapsed_time,
        }

        return obs, reward, terminated, truncated, info

    def get_raw_joint_positions(self) -> dict[str, float]:
        """Get raw joint positions in degrees."""
        return self._raw_joint_positions if self._raw_joint_positions is not None else {}

    def close(self):
        """Close the environment and disconnect from robot."""
        logging.info("[RobotEnv] Closing environment...")
        if hasattr(self.robot, "disconnect"):
            self.robot.disconnect()
        if self.teleop_device is not None and hasattr(self.teleop_device, "disconnect"):
            self.teleop_device.disconnect()
        super().close()


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
    complementary_data= processed_action_transition[TransitionKey.COMPLEMENTARY_DATA].copy()
    new_info = processed_action_transition[TransitionKey.INFO].copy()
    new_info.update(info)

    new_transition = create_transition(
        observation=obs,
        action=action,
        reward=reward,
        done=terminated,
        truncated=truncated,
        info=new_info,
        complementary_data=complementary_data
    )
    new_transition = env_processor(new_transition)

    return new_transition

def make_processors(
        env: gym.Env, teleop_device: Teleoperator | None, cfg: HILSerlRobotEnvConfig, device: str ="cpu"
) -> tuple[DataProcessorPipeline[EnvTransition, EnvTransition], DataProcessorPipeline[EnvTransition, EnvTransition]]:
    """Create action processors"""
    terminate_on_success = (
        cfg.processor.reset.terminate_on_success if cfg.processor.reset is not None else True
    )

    joint_names = hc_joint_names
    print(f"Joint names are {joint_names}")

    # Set up kinematics solver if inverse kinematics is configured
    kinematics_solver = None
    if cfg.processor.inverse_kinematics is not None:
        kinematics_solver = RobotKinematics(
            urdf_path=cfg.processor.inverse_kinematics.urdf_path,
            target_frame_name=cfg.processor.inverse_kinematics.target_frame_name,
            joint_names=joint_names,
        )

    if cfg.name == "gym_isaac_sim_hil":
        action_pipeline_steps = [
            AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
            AddTeleopEventsAsInfoStep(teleop_device=teleop_device),
            InterventionActionProcessorStep(
                use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
                terminate_on_success=terminate_on_success),
        ]

        # Replace InverseKinematicsProcessor with new kinematic processors
        if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
            # Add EE bounds and safety processor
            inverse_kinematics_steps = [
                MapTensorToDeltaActionDictStep(
                    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
                ),
                MapDeltaActionToRobotActionStep(),
                EEReferenceAndDelta(
                    kinematics=kinematics_solver,
                    end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                    motor_names=joint_names,
                    use_latched_reference=False,
                    use_ik_solution=True,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
                ),
                GripperVelocityToJoint(
                    clip_max=cfg.processor.max_gripper_pos,
                    speed_factor=1.0,
                    discrete_gripper=True,
                ),
                InverseKinematicsRLStep(
                    kinematics=kinematics_solver, motor_names=joint_names, initial_guess_current_joints=False
                ),
            ]
            action_pipeline_steps.extend(inverse_kinematics_steps)
            action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=joint_names))
            action_pipeline_steps.append(AddBatchDimensionProcessorStep())
            action_pipeline_steps.append(Degrees2RadiansActionProcessorStep())

        env_pipeline_steps = [
            Radians2DegreesObservationProcessor(),
            VanillaObservationProcessorStep()]


        # TODO Add Joint velocities
        # TODO Add motor current

        if kinematics_solver is not None:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=kinematics_solver,
                    motor_names=joint_names,
                )
        )
        env_pipeline_steps.append(
            AddProcessorObservationsToState()
        )

        if cfg.processor.image_preprocessing is not None:
            env_pipeline_steps.append(
                ImageCropResizeProcessorStep(
                    crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                    resize_size=cfg.processor.image_preprocessing.resize_size,
                )
            ) 
        
        # TODO Consider adding time limit processor

        # Add gripper penalty processor if gripper config exists and enabled
        if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
            env_pipeline_steps.append(
                GripperPenaltyProcessorStep(
                    penalty=cfg.processor.gripper.gripper_penalty,
                    max_gripper_pos=cfg.processor.max_gripper_pos,
                )
            )

        #note: Skipped reward classifier
        

        env_pipeline_steps.append(AddBatchDimensionProcessorStep())
        env_pipeline_steps.append(DeviceProcessorStep(device=device))
    
        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition)

    if cfg.name == "real_robot":
        action_pipeline_steps = [
            AddTeleopActionAsComplimentaryDataStep(teleop_device=teleop_device),
            AddTeleopEventsAsInfoStep(teleop_device=teleop_device)]

        # Convert teleop_action from joint space to delta x, y, z if kinematics solver is available
        if kinematics_solver is not None:
            action_pipeline_steps.append(
                TeleopConvertJointToDeltaStep(
                    kinematics=kinematics_solver,
                    motor_names=joint_names,
                    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
                )
            )
        action_pipeline_steps.append(
            InterventionActionProcessorStep(
                use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False,
                terminate_on_success=terminate_on_success),
        )

        # Replace InverseKinematicsProcessor with new kinematic processors
        if cfg.processor.inverse_kinematics is not None and kinematics_solver is not None:
            # Add EE bounds and safety processor
            inverse_kinematics_steps = [
                MapTensorToDeltaActionDictStep(
                    use_gripper=cfg.processor.gripper.use_gripper if cfg.processor.gripper is not None else False
                ),
                MapDeltaActionToRobotActionStep(),
                EEReferenceAndDelta(
                    kinematics=kinematics_solver,
                    end_effector_step_sizes=cfg.processor.inverse_kinematics.end_effector_step_sizes,
                    motor_names=joint_names,
                    use_latched_reference=False,
                    use_ik_solution=True,
                ),
                EEBoundsAndSafety(
                    end_effector_bounds=cfg.processor.inverse_kinematics.end_effector_bounds,
                ),
                GripperVelocityToJoint(
                    clip_max=cfg.processor.max_gripper_pos,
                    speed_factor=1.0,
                    discrete_gripper=True,
                ),
                InverseKinematicsRLStep(
                    kinematics=kinematics_solver, motor_names=joint_names, initial_guess_current_joints=False
                ),
            ]
            action_pipeline_steps.extend(inverse_kinematics_steps)
            action_pipeline_steps.append(RobotActionToPolicyActionProcessorStep(motor_names=joint_names))
            # action_pipeline_steps.append(AddBatchDimensionProcessorStep())

        env_pipeline_steps = [
            VanillaObservationProcessorStep()]


        # TODO Add Joint velocities
        # TODO Add motor current

        if kinematics_solver is not None:
            env_pipeline_steps.append(
                ForwardKinematicsJointsToEEObservation(
                    kinematics=kinematics_solver,
                    motor_names=joint_names,
                )
        )
        env_pipeline_steps.append(
            AddProcessorObservationsToState()
        )

        if cfg.processor.image_preprocessing is not None:
            env_pipeline_steps.append(
                ImageCropResizeProcessorStep(
                    crop_params_dict=cfg.processor.image_preprocessing.crop_params_dict,
                    resize_size=cfg.processor.image_preprocessing.resize_size,
                )
            ) 
        
        # TODO Consider adding time limit processor

        # Add gripper penalty processor if gripper config exists and enabled
        if cfg.processor.gripper is not None and cfg.processor.gripper.use_gripper:
            env_pipeline_steps.append(
                GripperPenaltyProcessorStep(
                    penalty=cfg.processor.gripper.gripper_penalty,
                    max_gripper_pos=cfg.processor.max_gripper_pos,
                )
            )

        #note: Skipped reward classifier
        

        env_pipeline_steps.append(AddBatchDimensionProcessorStep())
        env_pipeline_steps.append(DeviceProcessorStep(device=device))
    
        return DataProcessorPipeline(
            steps=env_pipeline_steps, to_transition=identity_transition, to_output=identity_transition
        ), DataProcessorPipeline(steps=action_pipeline_steps, to_transition=identity_transition, to_output=identity_transition)

    raise NotImplementedError(f"Processors for {cfg.name} not implemented")