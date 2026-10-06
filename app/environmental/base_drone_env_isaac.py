"""Isaac Lab DirectRLEnv port of BaseDroneEnv; registers "Isaac-Base-Drone-Direct-v0". Isaac venv only, see docs.md."""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from app.environmental.base_drone_env import ANG_VEL_SCALE, DIST_SCALE, HEIGHT_CLIP, POSITION_MODES, POS_SCALE, VEL_SCALE
from app.environmental.base_drone_env import MAX_RPM as OBS_MAX_RPM
from app.reward_functions.rewards import (
    APPROACH_MILESTONE_BUDGET,
    ATTITUDE_PENALTY,
    ATTITUDE_PITCH_DEG,
    ATTITUDE_ROLL_DEG,
    HIT_REWARD,
    HIT_THRESHOLD,
    INNER_ZONE_EXIT_PENALTY,
    INNER_ZONE_RADIUS,
    MILESTONE_BONUSES,
    MILESTONE_FRACS,
    OOB_PENALTY,
    OOB_RADIUS,
    OUTER_ZONE_EXIT_PENALTY,
    OUTER_ZONE_RADIUS,
    STABILITY_COEF,
    TARGET_FRACTION,
)
from app.dynamics import torch_methods as tp

# Never raise PHYSICS_DT to make training faster -- use `decimation` instead.
PHYSICS_DT = 1.0 / 240.0
MAX_POLICY_STEPS = 15_000

CROSS_SEC_AREA = 0.05
AIR_DENS = 1.225
K_WIND_COEFF = 0.1

# Feeds BaseDroneEnvIsaacCfg.action_space's class-body Box bounds -- keep in sync with robot.mass below.
_HOVER_THRUST = 1.5 * 9.81


def _symlog(x: torch.Tensor, linthresh: float) -> torch.Tensor:
    """Matches app.environmental.base_drone_env._symlog_scale exactly."""
    ax = x.abs()
    linear = x / linthresh
    log_part = torch.sign(x) * (1.0 + torch.log(torch.clamp(ax / linthresh, min=1e-8)))
    return torch.where(ax <= linthresh, linear, log_part)


class BaseDroneEnvIsaacWindow(BaseEnvWindow):
    """Window manager for the Isaac base-drone environment (debug target-marker toggle)."""

    def __init__(self, env: BaseDroneEnvIsaac, window_name: str = "IsaacLab"):
        super().__init__(env, window_name)
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BaseDroneEnvIsaacCfg(DirectRLEnvCfg):
    episode_length_s = MAX_POLICY_STEPS * PHYSICS_DT  # 62.5s, independent of decimation
    decimation = 4  # 4 physics substeps per policy step (60Hz effective control rate)
    # must be a bounded Box: an int gives an unbounded Box and NaNs the tanh rescale
    action_space: gym.spaces.Box = gym.spaces.Box(
        low=np.array([-_HOVER_THRUST, -0.5, -0.5, -0.5], dtype=np.float32),
        high=np.array([_HOVER_THRUST, 0.5, 0.5, 0.5], dtype=np.float32),
    )
    observation_space = 23  # matches build_observation's 23-dim layout
    state_space = 0
    debug_vis = True

    ui_window_class_type = BaseDroneEnvIsaacWindow

    sim: SimulationCfg = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=decimation,
        physx=sim_utils.PhysxCfg(enable_external_forces_every_iteration=True),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # 4096 is just the cfg's fallback default; callers override via parse_env_cfg(num_envs=N).
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    )

    # procedural RigidObject sized so its inertia ratio matches QuadConfig (docs.md "Robot asset")
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.4, 0.02),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=10.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )

    # QuadConfig equivalent -- BaseDroneEnv.reset() rebuilds an identical QuadConfig every episode.
    mass = 1.5
    inertia = (0.02, 0.02, 0.04)
    arm_length = 0.22
    drag_coeff = 0.035
    max_rpm = 12000.0
    motor_tau = 0.05


class BaseDroneEnvIsaac(DirectRLEnv):
    cfg: BaseDroneEnvIsaacCfg

    def __init__(self, cfg: BaseDroneEnvIsaacCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._t_cfg = tp.build_quad_config(
            mass=cfg.mass, inertia=cfg.inertia, arm_length=cfg.arm_length, drag_coeff=cfg.drag_coeff,
            max_rpm=cfg.max_rpm, motor_tau=cfg.motor_tau, device=self.device, dtype=torch.float32,
        )
        self._hover_thrust = self._t_cfg.mass * 9.81
        self._action_low = torch.as_tensor(self.single_action_space.low, device=self.device)
        self._action_high = torch.as_tensor(self.single_action_space.high, device=self.device)

        actual_mass = self._robot.root_physx_view.get_masses()
        actual_inertia = self._robot.root_physx_view.get_inertias()
        print(f"[BaseDroneEnvIsaac] robot mass={actual_mass.flatten()[:1].item():.6f} "
              f"(target {self._t_cfg.mass}), inertia diag={actual_inertia.flatten()[[0,4,8]]} "
              f"(target {self._t_cfg.inertia.cpu().tolist()})")

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._w_target = torch.zeros(self.num_envs, 4, device=self.device)
        self._rotor_rpm = torch.zeros(self.num_envs, 4, device=self.device)
        self._wind_vector = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self._target_pairs_start = None
        self._target_pairs_target = None
        self._target_pairs_yaw = None
        self._spawn_speed_range = None  # see set_spawn_speed_range
        self._obs_position_mode = "full"  # see set_obs_position_mode
        self._target_dist_range = None  # see set_target_distance_range

        self._start_dist = torch.ones(self.num_envs, device=self.device)
        self._prev_distance = torch.ones(self.num_envs, device=self.device)
        self._prev_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._milestones_hit = torch.zeros(self.num_envs, len(MILESTONE_FRACS), dtype=torch.bool, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["reward", "distance_to_goal"]
        }
        self._cached_reward = torch.zeros(self.num_envs, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = torch.clamp(actions, self._action_low, self._action_high)
        real_action = self._actions.clone()
        real_action[:, 0] = real_action[:, 0] + self._hover_thrust
        self._w_target = tp.mixer_inversion(self._t_cfg, real_action)

    def _apply_action(self):
        self._rotor_rpm = tp.motor_lag(self._rotor_rpm, self._w_target, self._t_cfg.motor_tau, PHYSICS_DT)

        net_thrust = tp.net_combining_thrust(self._t_cfg, self._rotor_rpm)
        net_torque = tp.net_combining_torque(self._t_cfg, self._rotor_rpm)

        vel_w = self._robot.data.root_lin_vel_w
        drag_w = tp.drag_force(vel_w, self._t_cfg.drag_coeff, CROSS_SEC_AREA, AIR_DENS)
        wind_w = tp.wind_force(self._wind_vector, self._t_cfg.mass, K_WIND_COEFF)

        # Torque is body-frame already; drag+wind are rotated from world into body frame to match.
        quat_wxyz = self._robot.data.root_quat_w
        force_body = torch.zeros(self.num_envs, 3, device=self.device)
        force_body[:, 2] = net_thrust
        force_body = force_body + math_utils.quat_apply_inverse(quat_wxyz, drag_w + wind_w)

        if getattr(self, "_debug_apply_action_once", False):
            print(f"[DEBUG _apply_action] net_thrust={net_thrust}  net_torque={net_torque}  "
                  f"force_body={force_body}  quat_wxyz={quat_wxyz}  "
                  f"body_mass={self._robot.root_physx_view.get_masses()}  "
                  f"body_inertia={self._robot.root_physx_view.get_inertias()}")
            self._debug_apply_action_once = False

        self._robot.permanent_wrench_composer.set_forces_and_torques(
            forces=force_body.unsqueeze(1),
            torques=net_torque.unsqueeze(1),
        )

    def _get_observations(self) -> dict:
        """Matches build_observation's 23-dim layout, reordering Isaac's (w,x,y,z) quat to scipy's (x,y,z,w)."""
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        vel = self._robot.data.root_lin_vel_w
        quat_wxyz = self._robot.data.root_quat_w
        quat_xyzw = torch.cat([quat_wxyz[:, 1:], quat_wxyz[:, 0:1]], dim=-1)
        ang_vel = self._robot.data.root_ang_vel_b
        target_local = self._desired_pos_w - self._terrain.env_origins
        rel = target_local - pos
        dist = torch.linalg.norm(rel, dim=-1, keepdim=True)

        _, _, yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
        yaw_err = (self._desired_yaw_w - yaw + math.pi) % (2 * math.pi) - math.pi

        pos_feat = _symlog(pos, POS_SCALE)
        if self._obs_position_mode == "height":
            # no absolute horizontal position, altitude clipped -- same layout/dim as "full" (see base_drone_env.POSITION_MODES)
            pos_feat = torch.zeros_like(pos_feat)
            pos_feat[:, 2] = _symlog(pos[:, 2].clamp(0.0, HEIGHT_CLIP), POS_SCALE)

        obs = torch.cat(
            [
                pos_feat,
                vel / VEL_SCALE,
                quat_xyzw,
                ang_vel / ANG_VEL_SCALE,
                self._rotor_rpm / OBS_MAX_RPM,
                _symlog(rel, DIST_SCALE),
                _symlog(dist, DIST_SCALE),
                torch.stack([torch.sin(yaw_err), torch.cos(yaw_err)], dim=-1),
            ],
            dim=-1,
        )
        return {"policy": obs}

    def _compute_dones_and_reward(self):
        """Shared by _get_dones/_get_rewards -- ports reward_func exactly (see rewards.py)."""
        pos = self._robot.data.root_pos_w - self._terrain.env_origins
        quat_wxyz = self._robot.data.root_quat_w
        roll, pitch, _yaw = math_utils.euler_xyz_from_quat(quat_wxyz)
        target_local = self._desired_pos_w - self._terrain.env_origins
        dist = torch.linalg.norm(target_local - pos, dim=-1)

        oob_radius = torch.clamp(self._start_dist * 3.0, min=OOB_RADIUS)
        oob = torch.isnan(pos).any(dim=-1) | (pos[:, 2] < 0.0) | (torch.linalg.norm(pos, dim=-1) > oob_radius)
        roll_bad = roll.abs() > math.radians(ATTITUDE_ROLL_DEG)
        pitch_bad = pitch.abs() > math.radians(ATTITUDE_PITCH_DEG)
        terminal_hard = oob | roll_bad | pitch_bad
        term_penalty = (
            oob.float() * OOB_PENALTY + roll_bad.float() * ATTITUDE_PENALTY + pitch_bad.float() * ATTITUDE_PENALTY
        )

        hit = dist < HIT_THRESHOLD

        diff_vec = target_local - pos
        prev_diff_vec = target_local - self._prev_position
        phi_now = -torch.sum(torch.abs(diff_vec), dim=-1) / self._start_dist
        phi_prev = -torch.sum(torch.abs(prev_diff_vec), dim=-1) / self._start_dist
        shaping = phi_now - phi_prev

        # Divides by max_episode_length (this env's real, fixed cap), not a per-distance value -- see rewards.py.
        step_penalty = -(TARGET_FRACTION * APPROACH_MILESTONE_BUDGET) / self.max_episode_length

        progress = 1.0 - dist / self._start_dist
        milestone_bonus = torch.zeros_like(dist)
        for idx, (frac, val) in enumerate(zip(MILESTONE_FRACS, MILESTONE_BONUSES)):
            newly_hit = (progress >= frac) & (~self._milestones_hit[:, idx])
            milestone_bonus = milestone_bonus + newly_hit.float() * val
            self._milestones_hit[:, idx] = self._milestones_hit[:, idx] | newly_hit

        # Anti-oscillation terms -- see rewards.py's OUTER_ZONE_RADIUS comment for the full derivation.
        vel = self._robot.data.root_lin_vel_w
        tilt = roll.abs() + pitch.abs()
        stability_penalty = torch.where(
            dist < OUTER_ZONE_RADIUS,
            -STABILITY_COEF * (torch.linalg.norm(vel, dim=-1) + tilt),
            torch.zeros_like(dist),
        )

        prev_dist = self._prev_distance
        was_in_inner = prev_dist < INNER_ZONE_RADIUS
        was_in_outer = prev_dist < OUTER_ZONE_RADIUS
        now_in_inner = dist < INNER_ZONE_RADIUS
        now_in_outer = dist < OUTER_ZONE_RADIUS
        exited_inner = was_in_inner & ~now_in_inner
        exited_outer = was_in_outer & ~now_in_outer & ~exited_inner  # elif equivalent
        exit_penalty = exited_inner.float() * INNER_ZONE_EXIT_PENALTY + exited_outer.float() * OUTER_ZONE_EXIT_PENALTY

        normal_reward = shaping + step_penalty + milestone_bonus + stability_penalty + exit_penalty
        reward = torch.where(terminal_hard, term_penalty, torch.where(hit, torch.full_like(dist, HIT_REWARD), normal_reward))

        self.extras["term_reasons"] = {"hit": hit, "oob": oob, "attitude_roll": roll_bad, "attitude_pitch": pitch_bad}

        continuing = ~(terminal_hard | hit)
        self._prev_position = torch.where(continuing.unsqueeze(-1), pos, self._prev_position)
        self._prev_distance = torch.where(continuing, dist, self._prev_distance)

        self._cached_reward = reward
        self._episode_sums["reward"] += reward
        self._episode_sums["distance_to_goal"] += dist
        return terminal_hard | hit

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = self._compute_dones_and_reward()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        """Snapshots obs before _reset_idx overwrites it, so a truncated env's terminal_observation is correct."""
        self.extras["terminal_observation"] = self._get_observations()["policy"]
        return self._cached_reward

    def set_obs_position_mode(self, mode: str):
        """"full" | "height"; must match the numpy env used for replay (run_config.json obs_position_mode)."""
        if mode not in POSITION_MODES:
            raise ValueError(f"mode must be one of {POSITION_MODES}, got {mode!r}")
        self._obs_position_mode = mode

    def set_target_pairs(self, pairs: list | None):
        """Pool of (start_pos, target_pos, target_yaw) sampled per reset; None clears it."""
        if pairs is None:
            self._target_pairs_start = None
            self._target_pairs_target = None
            self._target_pairs_yaw = None
            return
        starts = np.stack([np.asarray(p[0], dtype=np.float32) for p in pairs])
        targets = np.stack([np.asarray(p[1], dtype=np.float32) for p in pairs])
        yaws = np.array([float(p[2]) for p in pairs], dtype=np.float32)
        self._target_pairs_start = torch.as_tensor(starts, device=self.device)
        self._target_pairs_target = torch.as_tensor(targets, device=self.device)
        self._target_pairs_yaw = torch.as_tensor(yaws, device=self.device)

    def set_target_distance_range(self, low: float | None, high: float | None = None):
        """Fresh random target per reset (no pair pool): distance ~ U(low, high), uniform direction, random yaw."""
        self._target_dist_range = None if (low is None or high is None) else (float(low), float(high))

    def set_spawn_speed_range(self, low: float | None, high: float | None = None):
        """Spawn already moving toward the target at U(low, high) m/s (simulates a handoff); None = spawn at rest."""
        self._spawn_speed_range = None if (low is None or high is None) else (float(low), float(high))

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        n = len(env_ids)

        final_distance_to_goal = self._prev_distance[env_ids].mean()
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = torch.mean(self._episode_sums[key][env_ids]) / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        if self._target_pairs_start is not None:
            pool_idx = torch.randint(0, len(self._target_pairs_start), (n,), device=self.device)
            spawn_local = self._target_pairs_start[pool_idx]
            target_local = self._target_pairs_target[pool_idx]
            target_yaw = self._target_pairs_yaw[pool_idx]
            dist_mag = torch.linalg.norm(target_local - spawn_local, dim=-1)
        else:
            lo, hi = self._target_dist_range or (3.0, 10.0)
            dist_mag = torch.empty(n, device=self.device).uniform_(lo, hi)
            direction = torch.randn(n, 3, device=self.device)
            direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
            if self._target_dist_range is not None:
                # mirror underground directions so the 0.5 m floor clamp never changes the distance
                flip = (5.0 + direction[:, 2] * dist_mag) < 0.5
                direction[:, 2] = torch.where(flip, -direction[:, 2], direction[:, 2])
            spawn_local = torch.zeros(n, 3, device=self.device)
            spawn_local[:, 2] = 5.0
            target_local = spawn_local + direction * dist_mag.unsqueeze(-1)
            target_local[:, 2] = target_local[:, 2].clamp(min=0.5)
            target_yaw = torch.zeros(n, device=self.device)
            if self._target_dist_range is not None:
                target_yaw.uniform_(-math.pi, math.pi)

        self._desired_pos_w[env_ids] = target_local + self._terrain.env_origins[env_ids]
        self._desired_yaw_w[env_ids] = target_yaw

        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = spawn_local + self._terrain.env_origins[env_ids]
        default_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        default_root_state[:, 7:] = 0.0
        if self._spawn_speed_range is not None:
            # moving toward this episode's target, angular velocity stays 0
            low, high = self._spawn_speed_range
            speed = torch.empty(n, device=self.device).uniform_(low, high)
            to_target = target_local - spawn_local
            unit_dir = to_target / torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)
            default_root_state[:, 7:10] = unit_dir * speed.unsqueeze(-1)
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        hover_desired = torch.zeros(n, 4, device=self.device)
        hover_desired[:, 0] = self._hover_thrust
        self._rotor_rpm[env_ids] = tp.mixer_inversion(self._t_cfg, hover_desired)
        self._w_target[env_ids] = self._rotor_rpm[env_ids]
        self._actions[env_ids] = 0.0

        self._start_dist[env_ids] = dist_mag
        self._prev_distance[env_ids] = dist_mag
        self._prev_position[env_ids] = spawn_local
        self._milestones_hit[env_ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        self.goal_pos_visualizer.visualize(self._desired_pos_w)


gym.register(
    id="Isaac-Base-Drone-Direct-v0",
    entry_point=f"{__name__}:BaseDroneEnvIsaac",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:BaseDroneEnvIsaacCfg"},
)
