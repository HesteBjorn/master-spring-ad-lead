"""TFv6 PPO environment agent for CaRL custom leaderboard."""

# ruff: noqa: E402
from __future__ import annotations

import math
import os
import pathlib

import carla
import cv2
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np
import zmq

from rl_finetuning.tfv6_rl.path_utils import ensure_carl_paths

ensure_carl_paths()

from birds_eye_view.traffic_light import TrafficLightHandler
from leaderboard.autoagents import autonomous_agent
from leaderboard.autoagents.agent_wrapper import NextRoute
from leaderboard.envs.sensor_interface import SensorInterface
from leaderboard.utils import route_manipulation
from nav_planner import RoutePlanner as CarlaRoutePlanner
from reward.roach_reward import RoachReward
from reward.simple_reward import SimpleReward
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from lead.common import common_utils
from lead.common.base_agent import BaseAgent
from lead.common.pid_controller import LateralPIDController, PIDController, get_throttle
from lead.common.route_planner import RoutePlanner as LeadRoutePlanner
from lead.common.sensor_setup import av_sensor_setup
from lead.data_loader import carla_dataset_utils, training_cache
from lead.data_loader.carla_dataset_utils import rasterize_lidar
from lead.expert.config_expert import ExpertConfig
from lead.inference.config_closed_loop import ClosedLoopConfig
from rl_finetuning.tfv6_rl.action_codec import ActionCodec, infer_action_head_usage
from rl_finetuning.tfv6_rl.obs_codec import ObsCodec
from rl_finetuning.tfv6_rl.policy_tfv6_ppo import load_training_config
from rl_finetuning.tfv6_rl.privileged_measurements import (
    privileged_measurement_dim,
    validate_privileged_measurement_dim,
)
from rl_finetuning.tfv6_rl_config import GlobalConfig

jsonpickle_numpy.register_handlers()
jsonpickle.set_encoder_options("json", sort_keys=True, indent=4)


# Leaderboard function that selects the class used as agent.
def get_entry_point():
    return "EnvAgentTFv6"


class ClosedLoopController:
    def __init__(
        self,
        config_closed_loop: ClosedLoopConfig,
        config_expert: ExpertConfig,
        config_training,
    ) -> None:
        self.config_closed_loop = config_closed_loop
        self.config_expert = config_expert
        self.config_training = config_training

        self.lateral_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.turn_kp,
            k_i=self.config_closed_loop.turn_ki,
            k_d=self.config_closed_loop.turn_kd,
            n=self.config_closed_loop.turn_n,
        )
        self.longitudinal_waypoint_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )
        self.lateral_route_controller = LateralPIDController(self.config_closed_loop)
        self.longitudinal_target_speed_controller = PIDController(
            k_p=self.config_closed_loop.speed_kp,
            k_i=self.config_closed_loop.speed_ki,
            k_d=self.config_closed_loop.speed_kd,
            n=self.config_closed_loop.speed_n,
        )

    def execute_route_and_target_speed(
        self, pred_checkpoints, pred_target_speed, speed
    ):
        if isinstance(pred_checkpoints, np.ndarray):
            pred_checkpoints = pred_checkpoints[0]
        else:
            pred_checkpoints = pred_checkpoints[0].detach().cpu().numpy()
        speed = float(speed)
        if isinstance(pred_target_speed, np.ndarray):
            pred_target_speed = float(pred_target_speed.reshape(-1)[0])
        else:
            pred_target_speed = float(pred_target_speed)

        brake = bool(
            pred_target_speed < 0.01
            or (speed / pred_target_speed) > self.config_closed_loop.brake_ratio
        )
        steer = self.lateral_route_controller.step(
            pred_checkpoints,
            speed,
            0.0,
            0.0,
            sensor_agent_steer_correction=self.config_closed_loop.sensor_agent_steer_correction,
        )
        throttle, brake = get_throttle(
            brake, pred_target_speed, speed, self.config_expert
        )
        return steer, throttle, float(brake)

    def execute_waypoints(self, waypoints, speed):
        if isinstance(waypoints, np.ndarray):
            waypoints = waypoints[0]
        else:
            waypoints = waypoints[0].detach().cpu().numpy()
        speed = float(speed)

        one_second = int(
            self.config_training.carla_fps // self.config_training.data_save_freq
        )
        half_second = one_second // 2

        desired_speed = (
            np.linalg.norm(waypoints[half_second - 1] - waypoints[one_second - 1]) * 2.0
        )
        delta_speed = np.clip(
            desired_speed - speed, 0.0, self.config_closed_loop.wp_delta_clip
        )

        brake = (desired_speed < self.config_closed_loop.brake_speed) or (
            (speed / desired_speed) > self.config_closed_loop.brake_ratio
        )
        throttle = self.longitudinal_waypoint_controller.step(delta_speed)
        throttle = throttle if not brake else 0.0

        if self.config_closed_loop.tuned_aim_distance:
            aim_distance = np.clip(0.975532 * speed + 1.915288, 24, 105) / 10
        else:
            if desired_speed < self.config_closed_loop.aim_distance_threshold:
                aim_distance = self.config_closed_loop.aim_distance_slow
            else:
                aim_distance = self.config_closed_loop.aim_distance_fast

        aim_index = waypoints.shape[0] - 1
        for index, predicted_waypoint in enumerate(waypoints):
            if np.linalg.norm(predicted_waypoint) >= aim_distance:
                aim_index = index
                break

        aim = waypoints[aim_index]
        angle = np.degrees(np.arctan2(aim[1], aim[0])) / 90.0
        if speed < 0.01 or brake:
            angle = 0.0

        steer = self.lateral_waypoint_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        return float(steer), float(throttle), float(brake)


class EnvAgentTFv6(BaseAgent, autonomous_agent.AutonomousAgent):
    """Agent that streams TFv6 observations to PPO trainer and applies plan actions."""

    def __init__(self, carla_host, carla_port, debug=False):
        super().__init__(carla_host, carla_port, debug)
        track_name = os.environ.get("TFV6_RL_TRACK", "MAP_QUALIFIER")
        self.track = getattr(
            autonomous_agent.Track, track_name, autonomous_agent.Track.MAP_QUALIFIER
        )
        self.rl_config = GlobalConfig()
        self.training_config = None
        self.config_closed_loop = ClosedLoopConfig(raise_error_on_missing_key=False)
        self.config_expert = ExpertConfig()
        self.action_codec = None
        self.obs_codec = None
        self.controller = None
        self.command_waypoint_planners_dict = {}
        self.tp_global_plan_world_coord = None
        self.tp_downsample_factor = int(
            os.environ.get("TFV6_RL_TP_DOWNSAMPLE_FACTOR", "200")
        )

        self.initialized_global = False
        self.initialized_route = False
        self.send_first_observation = False

        self.num_send = 0
        self.last_input_data = None
        self.debug = int(os.environ.get("TFV6_RL_DEBUG", "0")) == 1

    def set_global_plan(self, global_plan_world_coord):
        self.dense_global_plan_world_coord = global_plan_world_coord
        self._global_plan_world_coord = global_plan_world_coord
        self.tp_global_plan_world_coord = global_plan_world_coord

        if self.tp_downsample_factor > 1:
            ds_ids = route_manipulation.downsample_route(
                global_plan_world_coord, self.tp_downsample_factor
            )
            self.tp_global_plan_world_coord = [
                (global_plan_world_coord[i][0], global_plan_world_coord[i][1])
                for i in ds_ids
            ]

        world = CarlaDataProvider.get_world()
        lat_ref, lon_ref = route_manipulation._get_latlon_ref(world)
        self._global_plan = route_manipulation.location_route_to_gps(
            global_plan_world_coord, lat_ref, lon_ref
        )

    def setup(self, exp_folder, port, route_config):
        # Recreate sensor interface per route to mirror wrapper lifecycle and
        # avoid stale sensor tags when routes restart.
        self.sensor_interface = SensorInterface()
        self.port = port
        self.exp_folder = exp_folder
        self.route_config = route_config
        self.step = -1
        self.termination = False
        self.truncation = False
        self.data = None
        self.last_timestamp = 0.0
        self.last_control = None
        # Route-local state must be reset each route. The leaderboard reuses the
        # same agent instance across route repetitions.
        self.initialized_route = False
        self.send_first_observation = False

        # Resolve TFv6 checkpoint. During debug runs this can be the agent-config folder
        # directly; for train_parallel-style launches it can be provided by TFV6_CHECKPOINT.
        checkpoint_dir = self._resolve_tfv6_checkpoint_dir(exp_folder)
        self.training_config = load_training_config(checkpoint_dir)
        use_route, use_waypoints, use_target_speed = infer_action_head_usage(
            self.training_config,
            steer_modality=self.config_closed_loop.steer_modality,
            throttle_modality=self.config_closed_loop.throttle_modality,
            brake_modality=self.config_closed_loop.brake_modality,
        )
        self.action_codec = ActionCodec(
            self.training_config,
            use_route=use_route,
            use_waypoints=use_waypoints,
            use_target_speed=use_target_speed,
        )
        # Build a provisional schema immediately so route restarts never observe
        # a None codec; it is rebuilt from synced trainer config in
        # agent_global_init() before the first packed rollout frame.
        self.obs_codec = ObsCodec(self.training_config, rl_config=self.rl_config)
        self.controller = ClosedLoopController(
            self.config_closed_loop, self.config_expert, self.training_config
        )

        super().setup(sensor_agent=True)

    def _resolve_tfv6_checkpoint_dir(self, exp_folder: str) -> str:
        candidates = [os.environ.get("TFV6_CHECKPOINT", None), exp_folder]
        checked = []

        for candidate in candidates:
            if candidate is None:
                continue
            candidate_path = pathlib.Path(candidate).expanduser()
            if not candidate_path.is_absolute():
                candidate_path = pathlib.Path.cwd() / candidate_path
            candidate_path = candidate_path.resolve()
            config_file = candidate_path / "config.json"
            if not config_file.exists():
                checked.append((str(candidate_path), "missing config.json"))
                continue

            has_model_file = any(candidate_path.glob("model*.pth"))
            if not has_model_file:
                checked.append((str(candidate_path), "missing model*.pth"))
                continue
            try:
                _ = load_training_config(str(candidate_path))
                return str(candidate_path)
            except Exception as exc:
                checked.append((str(candidate_path), f"invalid training config: {exc}"))
                continue

        checked_msg = ", ".join([f"{p} ({reason})" for p, reason in checked])
        raise FileNotFoundError(
            "Could not resolve TFv6 checkpoint folder. "
            "Set --agent-config to a TFv6 checkpoint folder or set TFV6_CHECKPOINT. "
            f"Checked: {checked_msg}"
        )

    def sensors(self):
        sensors = av_sensor_setup(
            config=self.training_config,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
            lidar=True,
            radar=self.training_config.use_radars,
            perturbate=False,
            sensor_agent=True,
        )
        if self.debug:
            type_counts = {}
            for sensor in sensors:
                sensor_type = sensor["type"]
                type_counts[sensor_type] = type_counts.get(sensor_type, 0) + 1
            print(
                f"[TFv6RL] track={self.track} sensor_counts={type_counts}",
                flush=True,
            )
        return sensors

    def agent_global_init(self):
        # Socket to talk to server
        print(f"Connecting to gymnasium server, port: {self.port}")
        self.context = zmq.Context()

        # Config socket
        conf_socket = self.context.socket(zmq.PAIR)
        comm_folder = pathlib.Path(__file__).parent / "comm_files"
        comm_folder.mkdir(parents=True, exist_ok=True)
        communication_file = comm_folder / str(self.port)
        conf_socket.connect(f"ipc://{communication_file}.conf_lock")
        json_config = conf_socket.recv_string()
        loaded_config = jsonpickle.decode(json_config)
        self.rl_config.__dict__.update(loaded_config.__dict__)
        self.obs_codec = ObsCodec(self.training_config, rl_config=self.rl_config)
        if bool(getattr(self.rl_config, "use_value_measurements", False)):
            validate_privileged_measurement_dim(self.rl_config)
        conf_socket.send_string(f"Config received port: {self.port}")

        # Connect to env gym to send observations
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.connect(f"ipc://{communication_file}.lock")
        self.socket.send_string(f"Connected to env_agent_tfV6 client. {self.port}")

        self.initialized_global = True

    def agent_route_init(self):
        self.vehicle = CarlaDataProvider.get_hero_actor()
        self.world = self.vehicle.get_world()
        settings = self.world.get_settings()
        assert math.isclose(
            settings.fixed_delta_seconds, 1.0 / self.rl_config.frame_rate
        )
        self.world_map = CarlaDataProvider.get_map()
        # SimpleReward's red-light criterion depends on TrafficLightHandler being initialized.
        TrafficLightHandler.reset(self.world, self.world_map)
        if self.debug:
            print(
                f"[TFv6RL] traffic_lights_initialized={TrafficLightHandler.num_tl}",
                flush=True,
            )

        if self.rl_config.reward_type == "roach":
            self.reward_handler = RoachReward(
                self.vehicle, self.world_map, self.world, self.rl_config
            )
        elif self.rl_config.reward_type == "simple_reward":
            self.reward_handler = SimpleReward(
                self.vehicle,
                self.world_map,
                self.world,
                self.rl_config,
                self.dense_global_plan_world_coord,
            )
        else:
            raise ValueError("Selected reward type is not implemented.")

        self.route_planner = CarlaRoutePlanner()
        self.route_planner.set_route(self.dense_global_plan_world_coord)
        self.command_waypoint_planners_dict = {}
        if not bool(getattr(self.training_config, "use_noisy_tp", False)):
            for dist in self.config_closed_loop.tp_distances:
                planner = LeadRoutePlanner(
                    dist, self.config_closed_loop.route_planner_max_distance
                )
                planner.set_route(self.tp_global_plan_world_coord, gps=False)
                self.command_waypoint_planners_dict[dist] = planner

        self.initialized_route = True

    def get_waypoint_route(self):
        pos = self.vehicle.get_transform().location
        pos = np.array([pos.x, pos.y])
        return self.route_planner.run_step(pos)

    def _tp_planners_dict(self):
        if bool(getattr(self.training_config, "use_noisy_tp", False)):
            return self.gps_waypoint_planners_dict
        return self.command_waypoint_planners_dict

    def _tp_ego_position(self, input_data: dict) -> np.ndarray:
        use_noisy_tp = bool(getattr(self.training_config, "use_noisy_tp", False))
        if use_noisy_tp:
            if bool(getattr(self.training_config, "use_kalman_filter_for_gps", True)):
                return np.array(self.filtered_state[:2], dtype=np.float32)
            return np.array(input_data["noisy_state"][:2], dtype=np.float32)

        # Match TFv6 training labels (pos_global) when noisy TP is disabled.
        ego_loc = self.vehicle.get_transform().location
        return np.array([ego_loc.x, ego_loc.y], dtype=np.float32)

    def _run_command_tp_planners(self, input_data: dict) -> None:
        if bool(getattr(self.training_config, "use_noisy_tp", False)):
            return
        if not self.command_waypoint_planners_dict:
            return
        ego_loc = self.vehicle.get_transform().location
        ego_xyz = np.array([ego_loc.x, ego_loc.y, input_data["noisy_state"][2]])
        for planner in self.command_waypoint_planners_dict.values():
            planner.run_step(ego_xyz)

    def _training_style_target_points(self) -> bool:
        # Match Base_TFv6 training TP semantics:
        # - use non-noisy TP coordinates
        # - use checkpoint TP pop distance
        return not bool(getattr(self.training_config, "use_noisy_tp", False))

    def set_target_points(self, input_data: dict, pop_distance: float) -> None:
        planner = self._tp_planners_dict()[pop_distance]

        next_target_points = [tp[0].tolist() for tp in planner.route]
        next_commands = [int(planner.route[i][1]) for i in range(len(planner.route))]

        # Merge duplicate consecutive target points
        filtered_tp_list = []
        filtered_command_list = []
        for pt, cmd in zip(next_target_points, next_commands, strict=False):
            if (
                len(next_target_points) == 2
                or not filtered_tp_list
                or not np.allclose(pt[:2], filtered_tp_list[-1][:2])
            ):
                filtered_tp_list.append(pt)
                filtered_command_list.append(cmd)
        next_target_points = filtered_tp_list
        next_commands = filtered_command_list

        def transform(point):
            ego_position = self._tp_ego_position(input_data)
            return common_utils.inverse_conversion_2d(
                np.array(point), np.array(ego_position), self.compass
            )

        if len(next_target_points) > 2:
            input_data["target_point_next"] = transform(next_target_points[2][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])
        else:
            assert len(next_target_points) == 2
            input_data["target_point_next"] = transform(next_target_points[1][:2])
            input_data["target_point"] = transform(next_target_points[1][:2])
            input_data["target_point_previous"] = transform(next_target_points[0][:2])

        input_data["command"] = carla_dataset_utils.command_to_one_hot(next_commands[0])
        input_data["next_command"] = carla_dataset_utils.command_to_one_hot(
            next_commands[1]
        )

    def preprocess_observation(self, input_data: dict) -> dict:
        # BaseAgent.tick mutates the dict in place; keep the caller-owned raw sensor dict untouched.
        input_data = dict(input_data)
        input_data = super().tick(
            input_data, use_kalman_filter=self.training_config.use_kalman_filter_for_gps
        )
        self._run_command_tp_planners(input_data)

        rgb = input_data["rgb"]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        _, rgb = cv2.imencode(
            ".jpg",
            rgb,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.config_closed_loop.jpeg_quality],
        )
        rgb = cv2.imdecode(rgb, cv2.IMREAD_UNCHANGED)
        rgb = np.transpose(rgb, (2, 0, 1))
        input_data["rgb"] = rgb

        if self.training_config.horizontal_fov_reduction > 0:
            crop_pixels = self.training_config.horizontal_fov_reduction
            if input_data["rgb"] is not None:
                _, h, w = input_data["rgb"].shape
                input_data["rgb"] = input_data["rgb"][:, :, crop_pixels:-crop_pixels]
                input_data["rgb"] = np.transpose(input_data["rgb"], (1, 2, 0))
                input_data["rgb"] = cv2.resize(
                    input_data["rgb"], (w, h), interpolation=cv2.INTER_LINEAR
                )
                input_data["rgb"] = np.transpose(input_data["rgb"], (2, 0, 1))

        # Cut cameras down to only used cameras
        if (
            self.training_config.num_used_cameras
            != self.training_config.num_available_cameras
        ):
            n = self.training_config.num_available_cameras
            w = input_data["rgb"].shape[2] // n
            rgb_slices = []
            for i, use in enumerate(self.training_config.used_cameras):
                if use:
                    s, e = i * w, (i + 1) * w
                    rgb_slices.append(input_data["rgb"][:, :, s:e])
            input_data["rgb"] = np.concatenate(rgb_slices, axis=2)

        if self._training_style_target_points():
            self.set_target_points(
                input_data, pop_distance=self.training_config.tp_pop_distance
            )
        else:
            self.set_target_points(
                input_data,
                pop_distance=self.config_closed_loop.route_planner_min_distance,
            )
            if self.config_closed_loop.sensor_agent_pop_distance_adaptive:
                dense_points = (
                    np.linalg.norm(
                        input_data["target_point"] - input_data["target_point_next"]
                    )
                    < 10.0
                    and min(
                        np.linalg.norm(input_data["target_point_previous"]),
                        np.linalg.norm(input_data["target_point"]),
                    )
                    < 10.0
                )
                dense_points = dense_points or (
                    np.linalg.norm(
                        input_data["target_point_previous"] - input_data["target_point"]
                    )
                    < 10.0
                    and min(
                        np.linalg.norm(input_data["target_point_previous"]),
                        np.linalg.norm(input_data["target_point"]),
                    )
                    < 10.0
                )
                if dense_points:
                    self.set_target_points(input_data, pop_distance=4.0)

            if (
                self.config_closed_loop.sensor_agent_skip_distant_target_point
                and np.linalg.norm(input_data["target_point_next"])
                > self.config_closed_loop.sensor_agent_skip_distant_target_point_threshold
            ):
                input_data["target_point_next"] = input_data["target_point"]

        # LiDAR accumulation
        lidar = self.accumulate_lidar()
        lidar = lidar[lidar[:, -1] < self.training_config.training_used_lidar_steps]

        input_data["rasterized_lidar"] = rasterize_lidar(
            config=self.training_config, lidar=lidar[:, :3]
        )[..., None]
        input_data["rasterized_lidar"] = training_cache.compress_float_image(
            input_data["rasterized_lidar"], self.training_config
        )
        input_data["rasterized_lidar"] = training_cache.decompress_float_image(
            input_data["rasterized_lidar"]
        ).squeeze()[None]

        if self.training_config.use_radars:
            input_data["radar"] = np.concatenate(
                carla_dataset_utils.preprocess_radar_input(
                    self.training_config, input_data
                ),
                axis=0,
            )

        obs = {
            "rgb": input_data["rgb"].astype(np.uint8),
            "rasterized_lidar": input_data["rasterized_lidar"].astype(np.float32),
            "target_point_previous": input_data["target_point_previous"].astype(
                np.float32
            ),
            "target_point": input_data["target_point"].astype(np.float32),
            "target_point_next": input_data["target_point_next"].astype(np.float32),
            "speed": np.array([input_data["speed"]], dtype=np.float32),
            "command": input_data["command"].astype(np.float32),
            "next_command": input_data["next_command"].astype(np.float32),
        }
        if self.training_config.use_radars:
            obs["radar"] = input_data["radar"].astype(np.float32)

        return obs

    def _build_privileged_measurements(
        self, timestamp: float, waypoint_route
    ) -> np.ndarray:
        if not bool(getattr(self.rl_config, "use_value_measurements", False)):
            return np.zeros((0,), dtype=np.float32)

        eval_time = float(max(getattr(self.rl_config, "eval_time", 1.0), 1e-6))
        remaining_time = (eval_time - float(timestamp)) / eval_time
        remaining_time = float(np.clip(remaining_time, 0.0, 1.0))

        block_detector = getattr(self.reward_handler, "block_detector", None)
        time_till_blocked = float(
            getattr(block_detector, "time_till_blocked", 1.0)
            if block_detector is not None
            else 1.0
        )
        time_till_blocked = float(np.clip(time_till_blocked, 0.0, 1.0))

        perc_route_left = float(len(waypoint_route)) / 100.0

        values: list[float] = [remaining_time, time_till_blocked, perc_route_left]

        if bool(getattr(self.rl_config, "use_ttc", False)):
            ttc_ticks = float(max(getattr(self.rl_config, "ttc_penalty_ticks", 1), 1))
            remaining_ttc = float(
                getattr(self.reward_handler, "remaining_ttc_penalty_ticks", 0.0)
            )
            values.append(float(np.clip(remaining_ttc / ttc_ticks, 0.0, 1.0)))

        if bool(getattr(self.rl_config, "use_comfort_infraction", False)):
            comfort_ticks = float(
                max(getattr(self.rl_config, "comfort_penalty_ticks", 1), 1)
            )
            comfort_arr = getattr(
                self.reward_handler, "remaining_comfort_penalty_ticks", None
            )
            if comfort_arr is None:
                values.extend([0.0] * 6)
            else:
                comfort_arr = np.asarray(comfort_arr, dtype=np.float32).reshape(-1)
                if comfort_arr.shape[0] != 6:
                    raise ValueError(
                        f"Expected 6 comfort penalty ticks, got {comfort_arr.shape[0]}"
                    )
                values.extend(np.clip(comfort_arr / comfort_ticks, 0.0, 1.0).tolist())

        expected_dim = privileged_measurement_dim(self.rl_config)
        if len(values) != expected_dim:
            raise ValueError(
                f"Privileged measurement length mismatch: built={len(values)} expected={expected_dim}"
            )
        return np.asarray(values, dtype=np.float32)

    def run_step(self, input_data, timestamp, sensors=None):
        self.step += 1
        self.last_timestamp = timestamp
        self.last_input_data = input_data

        if not self.initialized_global:
            self.agent_global_init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.last_control = control
            self.control = control
            return control

        if not self.initialized_route:
            self.agent_route_init()
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.last_control = control
            self.control = control
            return control

        if self.step < self.rl_config.start_delay_frames:
            control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)
            self.control = control
            return control

        if self.step % self.rl_config.action_repeat != 0:
            self.control = self.last_control
            return self.last_control

        waypoint_route = self.get_waypoint_route()
        obs = self.preprocess_observation(input_data)
        if self.debug and self.step == 0:
            obs_shapes = {k: v.shape for k, v in obs.items()}
            print(
                f"[TFv6RL] obs_shapes={obs_shapes}, action_dim={self.action_codec.action_dim}"
            )

        reward, termination, truncation, exploration_suggest = self.reward_handler.get(
            timestamp,
            waypoint_route,
            False,
            (),
            (),
            (),
            0.0,
        )
        if bool(getattr(self.rl_config, "use_value_measurements", False)):
            obs["privileged_measurements"] = self._build_privileged_measurements(
                timestamp, waypoint_route
            )

        data = {
            "observation": obs,
            "reward": reward,
            "termination": termination,
            "truncation": truncation,
            "info": exploration_suggest,
        }

        if termination or truncation:
            self.termination = termination
            self.truncation = truncation
            self.data = data
            raise NextRoute("Episode ended by reward.")

        self.num_send += 1
        packed_obs = self.obs_codec.pack(data["observation"])
        self.socket.send_multipart(
            (
                *packed_obs,
                np.array(data["reward"], dtype=np.float32),
                np.array(data["termination"], dtype=bool),
                np.array(data["truncation"], dtype=bool),
                np.array(data["info"]["n_steps"], dtype=np.int32),
                np.array(data["info"]["suggest"], dtype=np.int32),
                np.array(self.num_send, dtype=np.uint64),
            ),
            copy=False,
        )

        self.send_first_observation = True

        action = np.frombuffer(self.socket.recv(copy=False), dtype=np.float32)
        route, waypoints, target_speed = self.action_codec.decode(action)

        speed = obs["speed"][0]
        control = carla.VehicleControl(steer=0.0, throttle=0.0, brake=1.0)

        if route is not None and target_speed is not None:
            steer, throttle, brake = self.controller.execute_route_and_target_speed(
                route, target_speed, speed
            )
            control.steer = steer
            control.throttle = throttle
            control.brake = brake

        if waypoints is not None:
            wp_steer, wp_throttle, wp_brake = self.controller.execute_waypoints(
                waypoints, speed
            )
            if self.config_closed_loop.steer_modality == "waypoint":
                control.steer = wp_steer
            if self.config_closed_loop.throttle_modality == "waypoint":
                control.throttle = wp_throttle
            if self.config_closed_loop.brake_modality == "waypoint":
                control.brake = wp_brake

        if control.brake > 0.0:
            control.throttle = 0.0
            if speed < 0.01:
                control.steer = 0.0

        self.last_control = control
        self.control = control
        return control

    def destroy(self, results=None):
        if not self.send_first_observation:
            if hasattr(self, "reward_handler"):
                try:
                    self.reward_handler.destroy()
                except Exception:
                    pass
                del self.reward_handler
            return

        self.send_first_observation = False

        if self.termination or self.truncation:
            data = self.data
        else:
            waypoint_route = self.get_waypoint_route()
            if self.last_input_data is None:
                return
            obs = self.preprocess_observation(self.last_input_data)
            reward, termination, _, exploration_suggest = self.reward_handler.get(
                self.last_timestamp,
                waypoint_route,
                False,
                (),
                (),
                (),
                0.0,
            )
            if bool(getattr(self.rl_config, "use_value_measurements", False)):
                obs["privileged_measurements"] = self._build_privileged_measurements(
                    self.last_timestamp, waypoint_route
                )
            term = False
            trunc = True
            if termination:
                term = True
                trunc = False
            data = {
                "observation": obs,
                "reward": reward,
                "termination": term,
                "truncation": trunc,
                "info": exploration_suggest,
            }

        self.num_send += 1
        packed_obs = self.obs_codec.pack(data["observation"])
        self.socket.send_multipart(
            (
                *packed_obs,
                np.array(data["reward"], dtype=np.float32),
                np.array(data["termination"], dtype=bool),
                np.array(data["truncation"], dtype=bool),
                np.array(data["info"]["n_steps"], dtype=np.int32),
                np.array(data["info"]["suggest"], dtype=np.int32),
                np.array(self.num_send, dtype=np.uint64),
            ),
            copy=False,
        )
        if hasattr(self, "reward_handler"):
            try:
                self.reward_handler.destroy()
            except Exception:
                pass
            del self.reward_handler

        self.termination = False
        self.truncation = False
        self.initialized_route = False
