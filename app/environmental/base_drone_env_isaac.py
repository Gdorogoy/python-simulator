"""Isaac Lab DirectRLEnv port of BaseDroneEnv; registers "Isaac-Base-Drone-Direct-v0" (angle-style) and
"Isaac-Acro-Drone-Direct-v0" (acro-style). Isaac venv only, see docs.md."""

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

from app.control.los import CHASER_CRUISE_GAIN, chase_velocity, hold_velocity, velocity_to_action
from app.environmental.base_drone_env import ANG_VEL_SCALE, DIST_SCALE, HEIGHT_CLIP, POSITION_MODES, POS_SCALE, VEL_SCALE
from app.environmental.base_drone_env import MAX_RPM as OBS_MAX_RPM
from app.environmental.perception import ALLY_FEATURES, FLAG_FEATURES, OPPONENT_FEATURES, estimate_opponents, mode_flags
from app.reward_functions.angle_reward_rc import TIMER_FEATURES, zone_timer_features
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
from app.dynamics import angle_style_torch_methods as tp
from app.dynamics import drone_params as P
from app.dynamics.acro_style_methods import MAX_BODY_RATE
from app.dynamics.acro_style_torch_methods import TorchRateController

# Never raise PHYSICS_DT to make training faster -- use `decimation` instead.
PHYSICS_DT = 1.0 / 240.0
MAX_POLICY_STEPS = 15_000

CROSS_SEC_AREA = P.CROSS_SEC_AREA
AIR_DENS = P.AIR_DENS
K_WIND_COEFF = 0.1

# Per threat slot, appended AFTER the 23 base inputs: rel pos (3, symlog), rel vel (3), alive (1).
# Appending (never inserting) is what lets 23-input checkpoints load: see app.guidance.train.load_bc_checkpoint.
THREAT_FEATURES = 7
# PLACEHOLDER (angle drone only, nothing implemented): a threat the drone does not see right now may be remembered from the last fix or
# predicted by a filter. The velocity inputs (3 of the 7) are not needed by the simulator, only by the policy, so the layout may become 4 per
# slot (rel pos 3 + known 1) -- but that SHIFTS the indices after the first slot and old weights stop fitting, so it is not changed yet.
# The acro drone keeps its plain 23 inputs and one target.
# cfg.timer_obs adds TIMER_FEATURES (in_zone, zone_count / steps needed) after the threat slots -- again appended, never inserted.

# Threat drones: one colour per slot (the drone itself is red); parked/unused slots hold at THREAT_PARK_Z above their env origin.
THREAT_COLORS = ((0.10, 0.35, 0.95), (0.10, 0.80, 0.25), (0.95, 0.75, 0.10), (0.70, 0.25, 0.90), (0.10, 0.85, 0.85), (1.00, 0.50, 0.00))
THREAT_PARK_Z = 300.0

# Runner (acro moving target, cfg.runner): the target is a drone that flees. Appended AFTER every other input: its velocity relative
# to the drone (3, / VEL_SCALE). It waits (hovering on the target point) until the handoff, then flies straight away from the drone
# at its speed cap. No collision body: the hit is the reward's distance check, a body would only bounce the drone off it.
RUNNER_FEATURES = 3
RUNNER_COLOR = (1.0, 1.0, 1.0)

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
        low=np.array(P.angle_action_bounds(P.MASS)[0], dtype=np.float32),
        high=np.array(P.angle_action_bounds(P.MASS)[1], dtype=np.float32),
    )
    observation_space = 23  # matches build_observation's 23-dim layout; the trainer sets 23 + THREAT_FEATURES * max_threats
    max_threats = 0         # threat slots allocated (and observed) from the start; 0 = no threats, obs unchanged
    threat_radius = None    # m, collision-sphere radius of a threat drone; None = the drone's own body (same cuboid, same inertia)
    obs_layout = "v1"       # "v1": 23 base + 7/threat + timer (today); "v2": 23 base + timer + 4/opponent + 5/ally + 3 flags (docs.md "Observation layouts")
    max_allies = 0          # v2 only: ally slots (0 = 1v1, the section has zero width)
    timer_obs = False       # append the zone timer (reward_functions.angle_reward_rc.zone_timer_features); the trainer sets observation_space
    runner = False          # the target is a fleeing drone (RUNNER_FEATURES appended last); the trainer sets observation_space
    runner_radius = 0.25    # m, visual sphere of the runner (no collision)
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
    mass = P.MASS
    inertia = P.INERTIA
    arm_length = P.ARM_LENGTH
    drag_coeff = P.DRAG_COEFF
    max_rpm = P.MAX_RPM
    motor_tau = P.MOTOR_TAU

    # "angle": action = [thrust, roll/pitch/yaw torque]; "acro": [thrust, p/q/r body rate] (dynamics/docs.md)
    drone_style = "angle"


@configclass
class AcroDroneEnvIsaacCfg(BaseDroneEnvIsaacCfg):
    action_space: gym.spaces.Box = gym.spaces.Box(
        low=np.array([P.acro_thrust_bounds(P.MASS)[0], *(-r for r in MAX_BODY_RATE)], dtype=np.float32),
        high=np.array([P.acro_thrust_bounds(P.MASS)[1], *MAX_BODY_RATE], dtype=np.float32),
    )
    drone_style = "acro"


class BaseDroneEnvIsaac(DirectRLEnv):
    cfg: BaseDroneEnvIsaacCfg

    def __init__(self, cfg: BaseDroneEnvIsaacCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._sync_threat_bodies()
        self._sync_runner_body()
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

        self._acro = cfg.drone_style == "acro"
        if self._acro:
            self._rate_ctrl = TorchRateController(self._t_cfg, self.num_envs)
            self._thrust_cmd = torch.zeros(self.num_envs, device=self.device)
            self._rate_cmd = torch.zeros(self.num_envs, 3, device=self.device)

        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._desired_yaw_w = torch.zeros(self.num_envs, device=self.device)
        self._target_pairs_start = None
        self._target_pairs_target = None
        self._target_pairs_yaw = None
        self._spawn_speed_range = None  # see set_spawn_speed_range
        self._obs_position_mode = "full"  # see set_obs_position_mode
        self._target_dist_range = None  # see set_target_distance_range

        # ---- threats (interceptors) and allies; K slots are allocated up front, unused ones are masked by *_alive ----
        K = int(cfg.max_threats)
        self._K = K
        self.threat_pos_w = torch.zeros(self.num_envs, K, 3, device=self.device)
        self.threat_vel_w = torch.zeros(self.num_envs, K, 3, device=self.device)
        self.threat_alive = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)
        self._threat_fresh = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)   # needs (re)spawn
        self._threat_prev_rel = torch.zeros(self.num_envs, K, 3, device=self.device)
        A = int(cfg.max_allies)
        self.allied_pos_w = torch.zeros(self.num_envs, A, 3, device=self.device)   # ally slots (obs_layout "v2"); no ally behaviour yet
        self.allied_vel_w = torch.zeros(self.num_envs, A, 3, device=self.device)
        self.allied_alive = torch.zeros(self.num_envs, A, dtype=torch.bool, device=self.device)
        self._obs_v2 = cfg.obs_layout == "v2"
        if cfg.obs_layout not in ("v1", "v2"):
            raise ValueError(f"obs_layout must be 'v1' or 'v2', got {cfg.obs_layout!r}")
        self._threat_active_k = 0
        self._threat_max_speed = 10.0
        self._threat_min_speed = 10.0
        self._threat_speed_cap = torch.full((self.num_envs, K, 1), 10.0, device=self.device)   # per env and slot, drawn at spawn
        self._threat_spawn_min = 0.0
        self._threat_release_frac = 1.0                       # fraction of the handoff distance still LEFT when a threat is released
        self._threat_release_range = None                     # (lo, hi) of that fraction: each threat draws its own, per episode
        self._threat_release_frac_slot = torch.ones(self.num_envs, K, device=self.device)
        self._threat_released = torch.zeros(self.num_envs, K, dtype=torch.bool, device=self.device)   # per slot, latched
        self._threat_spawn_z = 3.0
        self._threat_spread = 30.0
        self._threat_spawn = "uniform"
        self._geom_weights = (1.0, 1.0)                       # spawn="geometry": head-on, tail (see _spawn_geometry)
        self._geom_reaction_s = 1.5
        self._geom_tail_gap = (15.0, 40.0)
        self._geom_jitter = 2.0
        self._threat_spawned = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)   # geometry spawn done this episode
        self._threat_geom = torch.full((self.num_envs, K), -1, dtype=torch.long, device=self.device)   # per slot: 0 head-on, 1 tail, -1 none
        self._threat_prob = 1.0                               # chance an episode has chasers at all (set_threats(threat_prob=))
        self._threat_on = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)   # this episode has its chasers (else every slot stays dead)
        self._threat_buffer = 0.5                             # base layout: extra clearance between neighbouring threats, m
        self._threat_gains = None                             # LOS attitude gains + tilt cap, see set_threat_gains
        # threat drones run the drone's own rotor model; the controller outputs angle-style actions whatever the drone's style is
        self._t_rpm = torch.zeros(self.num_envs, K, 4, device=self.device)
        self._t_w_target = torch.zeros(self.num_envs, K, 4, device=self.device)
        self._t_low = torch.as_tensor(P.angle_action_bounds(P.MASS)[0], dtype=torch.float32, device=self.device)
        self._t_high = torch.as_tensor(P.angle_action_bounds(P.MASS)[1], dtype=torch.float32, device=self.device)
        self._threat_hold_w = torch.zeros(self.num_envs, K, 3, device=self.device)   # where a waiting / parked threat hovers
        _hover = torch.zeros(1, 4, device=self.device)
        _hover[:, 0] = self._t_cfg.mass * 9.81
        self._t_hover_rpm = tp.mixer_inversion(self._t_cfg, _hover)[0]               # (4,) rotor speeds that hold the weight
        self._t_rpm[:] = self._t_hover_rpm
        self._t_w_target[:] = self._t_hover_rpm
        self._threat_policy = None   # callable(env) -> (N,K,3) desired velocity; None = limit-LOS chase (control/los.py)

        # ---- runner (cfg.runner): the target itself is a drone that flees after the handoff, see set_runner ----
        self._runner = bool(cfg.runner)
        N = self.num_envs
        self._runner_vel_w = torch.zeros(N, 3, device=self.device)        # the target's velocity (0 until released); obs + LOS teacher read it
        self._runner_released = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._runner_fresh = torch.zeros(N, dtype=torch.bool, device=self.device)
        self._runner_hold_w = torch.zeros(N, 3, device=self.device)
        self._runner_cap = torch.full((N, 1), 8.0, device=self.device)
        self._runner_speed = (8.0, 8.0)
        self._runner_escape_dist = 100.0
        self._runner_min_alt = 2.0
        self._runner_gains = None
        self._r_rpm = self._t_hover_rpm.expand(N, 4).clone()
        self._r_w_target = self._r_rpm.clone()
        self._prev_target_w = self._desired_pos_w.clone()   # target one step ago (rewards: true distance change of a moving target)

        self._start_dist = torch.ones(self.num_envs, device=self.device)
        self._prev_distance = torch.ones(self.num_envs, device=self.device)
        self._prev_position = torch.zeros(self.num_envs, 3, device=self.device)
        self._milestones_hit = torch.zeros(self.num_envs, len(MILESTONE_FRACS), dtype=torch.bool, device=self.device)

        # zone timer: the rc reward (install_reward_rc) advances these and sets the radius / limit; all zero under other rewards
        self._timer_obs = bool(cfg.timer_obs)
        self._zone_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._zone_entered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._zone_radius = 5.0
        self._zone_limit = 1
        # post-hit window (rc reward, cfg.post_hit_steps): hit already scored, and steps since
        self._hit_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._post_hit = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["reward", "distance_to_goal"]
        }
        self._cached_reward = torch.zeros(self.num_envs, device=self.device)

        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["robot"] = self._robot
        # one body per threat slot (self._K does not exist yet: this runs inside DirectRLEnv.__init__)
        self._threat_objs = []
        for k in range(int(self.cfg.max_threats)):
            obj = RigidObject(self._threat_cfg(k))
            self.scene.rigid_objects[f"threat{k}"] = obj
            self._threat_objs.append(obj)
        self._runner_obj = None
        if self.cfg.runner:
            base = self.cfg.robot.spawn
            spawn = sim_utils.SphereCfg(radius=float(self.cfg.runner_radius), mass_props=base.mass_props, rigid_props=base.rigid_props,
                                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=RUNNER_COLOR))
            self._runner_obj = RigidObject(self.cfg.robot.replace(prim_path="/World/envs/env_.*/Runner", spawn=spawn,
                                                                  init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, THREAT_PARK_Z))))
            self.scene.rigid_objects["runner"] = self._runner_obj
        size = self.cfg.robot.spawn.size
        self._threat_body_radius = float(self.cfg.threat_radius) if self.cfg.threat_radius is not None             else 0.5 * math.sqrt(size[0] ** 2 + size[1] ** 2 + size[2] ** 2)     # half the body diagonal: bounding sphere of the cuboid

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------- threats start
    # Threats are real drones: a RigidObject per slot, the drone's own rotor model (motor lag, mixer, drag, wind) stepped every physics
    # substep, flown by the limit-LOS controller in app/control/los.py. They collide (sphere of cfg.threat_radius, or the drone's own body).

    def _threat_cfg(self, k: int) -> RigidObjectCfg:
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=THREAT_COLORS[k % len(THREAT_COLORS)])
        base = self.cfg.robot.spawn
        if self.cfg.threat_radius is None:
            spawn = base.replace(visual_material=material)           # the drone's body: same cuboid, same inertia
        else:
            spawn = sim_utils.SphereCfg(radius=float(self.cfg.threat_radius), mass_props=base.mass_props, rigid_props=base.rigid_props,
                                        collision_props=base.collision_props, visual_material=material)
        return self.cfg.robot.replace(prim_path=f"/World/envs/env_.*/Threat{k}", spawn=spawn,
                                      init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0 * k, 0.0, THREAT_PARK_Z)))

    def _sync_threat_bodies(self):
        """A sphere has the wrong inertia for a drone: copy the drone's mass and inertia onto every threat body (PhysX view)."""
        if not self._threat_objs or self.cfg.threat_radius is None:
            return
        idx = self._robot._ALL_INDICES.cpu()
        masses, inertias = self._robot.root_physx_view.get_masses(), self._robot.root_physx_view.get_inertias()
        for obj in self._threat_objs:
            obj.root_physx_view.set_masses(masses.clone(), idx)
            obj.root_physx_view.set_inertias(inertias.clone(), idx)
        print(f"[BaseDroneEnvIsaac] {len(self._threat_objs)} threat drones, collision sphere r={self.cfg.threat_radius} m, "
              f"mass={self._threat_objs[0].root_physx_view.get_masses().flatten()[:1].item():.4f}, "
              f"inertia diag={self._threat_objs[0].root_physx_view.get_inertias().flatten()[[0, 4, 8]]}")

    def set_threat_gains(self, gains_by_dist: dict, max_tilt: float, cruise_gain: float = CHASER_CRUISE_GAIN):
        """Attitude gains for the threat drones: the SHORT-RANGE entry of the PID table (smallest distance key), the same set the LOS
        teacher uses. `max_tilt` (rad) is the airframe tilt clamp, not a chaser limit. Call once before stepping with threats."""
        g = gains_by_dist[min(gains_by_dist, key=lambda key: float(key))]
        self._threat_gains = dict(kp_att=float(g["kp_att"]), kd_att=float(g["kd_att"]), kp_yaw=float(g["kp_yaw"]), kd_yaw=float(g["kd_yaw"]),
                                  max_tilt=float(max_tilt), cruise_gain=float(cruise_gain))

    def set_threats(self, active: int, *, max_speed: float = 10.0, spawn_z: float = 3.0, spread: float = 30.0,
                    spawn: str = "uniform", spawn_min: float = 0.0, min_speed: float | None = None,
                    release_frac: float = 1.0, release_frac_range: tuple[float, float] | None = None,
                    buffer: float = 0.5, policy=None, geometry_weights: tuple[float, float] = (1.0, 1.0),
                    reaction_s: float = 1.5, tail_gap: tuple[float, float] = (15.0, 40.0), jitter: float = 2.0,
                    threat_prob: float = 1.0):
        """Switch `active` of the allocated slots on.
        spawn "uniform" ("sphere" is an alias): every threat starts at a random point of the ball spawn_min..spread around the base
        (uniform in volume). spawn "base": all start at one point, the base at spawn_z height; slot 0 on it, the others on a horizontal
        ring around it with at least 2*radius + buffer between any two bodies, so they do not spawn inside each other.
        Release: a threat waits (hovering) until the drone, flown by the policy, has this fraction of the handoff distance LEFT:
        release_frac fixed, or release_frac_range=(lo, hi) drawn per threat and episode (0.01, 0.67) = released between 33% and 99% of the
        way after the handoff. max_speed / min_speed: the commanded chase speed (per threat U(min, max)); the only limit on the chase.
        threat_prob: each episode has its chasers with this probability (drawn at reset); otherwise every slot stays dead (alive = 0, so the
        observation is all zeros there, like an unused slot) and nothing can catch the drone."""
        if active > self._K:
            raise ValueError(f"active={active} > allocated max_threats={self._K}")
        spawn = "uniform" if spawn == "sphere" else spawn
        if spawn not in ("uniform", "base", "geometry"):
            raise ValueError(f"spawn must be 'uniform', 'base' or 'geometry', got {spawn!r}")
        if len(geometry_weights) != 2 or min(geometry_weights) < 0 or sum(geometry_weights) <= 0:
            raise ValueError(f"geometry_weights = (head-on, tail) must be >= 0 and not all zero, got {geometry_weights}")
        if reaction_s <= 0 or jitter < 0 or not 0 <= tail_gap[0] <= tail_gap[1]:
            raise ValueError("geometry spawn: need reaction_s > 0, jitter >= 0, 0 <= tail_gap lo <= hi")
        min_speed = max_speed if min_speed is None else min_speed
        if not 0.0 <= spawn_min <= spread:
            raise ValueError(f"need 0 <= spawn_min <= spread, got {spawn_min} and {spread}")
        if not 0.0 < min_speed <= max_speed:
            raise ValueError(f"need 0 < min_speed <= max_speed, got {min_speed} and {max_speed}")
        if not 0.0 < release_frac <= 1.0:
            raise ValueError(f"release_frac must be in (0, 1], got {release_frac}")
        if release_frac_range is not None and not 0.0 < release_frac_range[0] <= release_frac_range[1] <= 1.0:
            raise ValueError(f"release_frac_range needs 0 < lo <= hi <= 1, got {release_frac_range}")
        if buffer < 0.0:
            raise ValueError(f"buffer must be >= 0, got {buffer}")
        if not 0.0 <= threat_prob <= 1.0:
            raise ValueError(f"threat_prob must be in [0, 1], got {threat_prob}")

        self._threat_prob = float(threat_prob)
        self._threat_spawn = spawn
        self._geom_weights = tuple(float(x) for x in geometry_weights)
        self._geom_reaction_s, self._geom_jitter = float(reaction_s), float(jitter)
        self._geom_tail_gap = (float(tail_gap[0]), float(tail_gap[1]))
        self._threat_active_k = int(active)
        self._threat_max_speed, self._threat_min_speed = float(max_speed), float(min_speed)
        self._threat_spawn_min, self._threat_release_frac = float(spawn_min), float(release_frac)
        self._threat_release_range = None if release_frac_range is None else (float(release_frac_range[0]), float(release_frac_range[1]))
        self._threat_spawn_z, self._threat_spread, self._threat_buffer = float(spawn_z), float(spread), float(buffer)
        self._threat_policy = policy
        self._threat_fresh[:] = True

    def _base_ring(self, n: int) -> torch.Tensor:
        """(n, K, 3) offsets for the "base" spawn: slot 0 at the centre, slots 1..K-1 on a horizontal ring (random rotation per env) whose
        radius keeps every neighbouring pair, and every ring-to-centre pair, at least gap = 2*body_radius + buffer apart:
        rho = max(gap, gap / (2 sin(pi/m))) for m = K-1 ring slots (chord between neighbours = 2 rho sin(pi/m))."""
        off = torch.zeros(n, self._K, 3, device=self.device)
        m = self._K - 1
        if m == 0:
            return off
        gap = 2.0 * self._threat_body_radius + self._threat_buffer
        rho = gap if m == 1 else max(gap, gap / (2.0 * math.sin(math.pi / m)))
        ang = torch.rand(n, 1, device=self.device) * (2.0 * math.pi) + torch.arange(m, device=self.device) * (2.0 * math.pi / m)
        off[:, 1:, 0], off[:, 1:, 1] = rho * torch.cos(ang), rho * torch.sin(ang)
        return off

    def _materialize_threats(self):
        """Spawn threats for envs that were just reset. Lazy on purpose: wrappers (the handoff stage) overwrite the target
        and drone pose after _reset_idx, and the threat spawn must see the final ones."""

        if self._K == 0 or not bool(self._threat_fresh.any()):
            return
        K = self._K
        ids = torch.nonzero(self._threat_fresh, as_tuple=False).squeeze(-1)
        n = len(ids)
        origins = self._terrain.env_origins[ids]
        base = self._desired_pos_w[ids].clone()
        if self._threat_spawn == "uniform":
            direction = torch.randn(n, K, 3, device=self.device)
            direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
            lo3, hi3 = self._threat_spawn_min ** 3, self._threat_spread ** 3
            radius = (lo3 + (hi3 - lo3) * torch.rand(n, K, 1, device=self.device)).pow(1.0 / 3.0)   # uniform in volume
            pos = base.unsqueeze(1) + direction * radius
            floor = origins[:, 2:3]
            pos[:, :, 2] = floor + (pos[:, :, 2] - floor).abs()          # mirror anything underground
        elif self._threat_spawn == "base":
            base[:, 2] = self._threat_spawn_z + origins[:, 2]
            pos = base.unsqueeze(1).repeat(1, K, 1) + self._base_ring(n)
        else:                                                   # "geometry": placed at the handoff, see _spawn_geometry
            pos = base.unsqueeze(1).repeat(1, K, 1)
        # slots beyond `active` are parked high above their env and hover there (never observed: threat_alive masks them)
        self._threat_on[ids] = torch.rand(n, device=self.device) < self._threat_prob               # chasers this episode?
        alive = (torch.arange(K, device=self.device).unsqueeze(0) < self._threat_active_k) & self._threat_on[ids].unsqueeze(1)   # (n, K)
        if self._threat_spawn == "geometry":
            alive = torch.zeros_like(alive)                     # nothing is there until the handoff
        park_off = torch.zeros(K, 3, device=self.device)
        park_off[:, 0] = 10.0 * torch.arange(K, device=self.device)
        park_off[:, 2] = THREAT_PARK_Z
        pos = torch.where(alive.unsqueeze(-1), pos, origins.unsqueeze(1) + park_off)

        quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
        zero_vel = torch.zeros(n, 6, device=self.device)
        for k, obj in enumerate(self._threat_objs):
            obj.write_root_pose_to_sim(torch.cat([pos[:, k], quat0], dim=-1), ids)
            obj.write_root_velocity_to_sim(zero_vel, ids)
        self._t_rpm[ids] = self._t_hover_rpm
        self._t_w_target[ids] = self._t_hover_rpm
        self.threat_pos_w[ids] = pos
        self.threat_vel_w[ids] = 0.0
        self._threat_hold_w[ids] = pos
        self._threat_speed_cap[ids] = self._threat_min_speed + (self._threat_max_speed - self._threat_min_speed) * \
            torch.rand(n, K, 1, device=self.device)
        if self._threat_release_range is None:
            self._threat_release_frac_slot[ids] = self._threat_release_frac
        else:
            lo, hi = self._threat_release_range
            self._threat_release_frac_slot[ids] = lo + (hi - lo) * torch.rand(n, K, device=self.device)
        self._threat_released[ids] = False
        self.threat_alive[ids] = alive.expand(n, K)
        self._threat_prev_rel[ids] = pos - self._robot.data.root_pos_w[ids].unsqueeze(1)
        self._threat_spawned[ids] = False
        self._threat_geom[ids] = -1
        self._threat_fresh[ids] = False

    GEOM_NAMES = ("head-on", "tail")
    BUCKET_NAMES = GEOM_NAMES + ("none",)      # threat_bucket(): the two geometries, then "no chaser this episode"

    def threat_bucket(self) -> torch.Tensor:
        """(N,) long: the first chaser's geometry (0 head-on, 1 tail), 2 for an episode without chasers, -1 while not decided
        yet (geometry spawn, before the handoff)."""
        return torch.where(self._threat_on, self._threat_geom[:, 0], torch.full_like(self._threat_geom[:, 0], len(self.GEOM_NAMES)))

    def _spawn_geometry(self):
        """spawn="geometry": at the handoff (the step after the model takes over; the first step when there is no handoff stage) put every
        active chaser at a position relative to the DRONE, already moving, and release it at once. Each chaser draws its own geometry with
        geometry_weights = (head-on, tail):
          head-on : ahead along the drone's velocity at max(20 m, reaction_s * (drone speed + chaser cap)), so a dodge is possible,
                    +-jitter m off the line, flying back at the drone at its speed cap;
          tail    : U(tail_gap) m behind the drone, flying forward at its speed cap.
        Every position is kept at least 1 m above the ground. Slots beyond `active` stay parked."""
        N, K, dev = self.num_envs, self._K, self.device
        handed = getattr(self, "_handed_off", None)
        flying = torch.ones(N, dtype=torch.bool, device=dev) if handed is None else handed
        ids = torch.nonzero(flying & ~self._threat_spawned, as_tuple=False).squeeze(-1)
        n = ids.numel()
        if n == 0:
            return
        drone_p = self._robot.data.root_pos_w[ids]
        drone_v = self._robot.data.root_lin_vel_w[ids]
        speed = torch.linalg.norm(drone_v, dim=-1, keepdim=True)
        to_target = self._desired_pos_w[ids] - drone_p
        to_target = to_target / torch.linalg.norm(to_target, dim=-1, keepdim=True).clamp_min(1e-6)
        fwd = torch.where(speed > 1.0, drone_v / speed.clamp_min(1e-6), to_target)                # direction of travel (n, 3)
        up = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        side = torch.cross(fwd, up, dim=-1)
        side_norm = torch.linalg.norm(side, dim=-1, keepdim=True)
        side = torch.where(side_norm > 1e-3, side / side_norm.clamp_min(1e-6), torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
        w = torch.tensor(self._geom_weights, device=dev, dtype=torch.float32)
        geom = torch.multinomial(w / w.sum(), n * K, replacement=True).view(n, K)                 # 0 head-on, 1 tail
        cap = self._threat_speed_cap[ids]                                                         # (n, K, 1)
        f, s_, u_ = fwd.unsqueeze(1), side.unsqueeze(1), up.unsqueeze(1)

        def rnd(lo, hi):
            return lo + (hi - lo) * torch.rand(n, K, 1, device=dev)

        j = self._geom_jitter
        head_on = f * torch.clamp((speed.unsqueeze(1) + cap) * self._geom_reaction_s, min=20.0) + s_ * rnd(-j, j) + u_ * rnd(-j, j)
        tail = -f * rnd(*self._geom_tail_gap)
        g = geom.unsqueeze(-1)
        offset = torch.where(g == 0, head_on, tail)
        origins = self._terrain.env_origins[ids]
        pos = drone_p.unsqueeze(1) + offset
        pos[..., 2] = torch.maximum(pos[..., 2], origins[:, 2:3] + 1.0)
        # every chaser starts moving AT the drone at its cap, i.e. exactly what the limit-LOS controller will command, so there is no
        # velocity error to fight at the spawn
        to_drone = drone_p.unsqueeze(1) - pos
        vel0 = to_drone / torch.linalg.norm(to_drone, dim=-1, keepdim=True).clamp_min(1e-6) * cap
        active = (torch.arange(K, device=dev).unsqueeze(0) < self._threat_active_k) & self._threat_on[ids].unsqueeze(1)   # (n, K)
        park_off = torch.zeros(K, 3, device=dev)
        park_off[:, 0] = 10.0 * torch.arange(K, device=dev)
        park_off[:, 2] = THREAT_PARK_Z
        pos = torch.where(active.unsqueeze(-1), pos, origins.unsqueeze(1) + park_off)
        vel0 = torch.where(active.unsqueeze(-1), vel0, torch.zeros_like(vel0))

        quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand(n, 4)
        zero_w = torch.zeros(n, 3, device=dev)
        for k, obj in enumerate(self._threat_objs):
            obj.write_root_pose_to_sim(torch.cat([pos[:, k], quat0], dim=-1), ids)
            obj.write_root_velocity_to_sim(torch.cat([vel0[:, k], zero_w], dim=-1), ids)
        self._t_rpm[ids] = self._t_hover_rpm
        self._t_w_target[ids] = self._t_hover_rpm
        self.threat_pos_w[ids] = pos
        self.threat_vel_w[ids] = vel0
        self._threat_hold_w[ids] = pos
        self._threat_released[ids] = True
        self.threat_alive[ids] = active.expand(n, K)
        self._threat_prev_rel[ids] = pos - drone_p.unsqueeze(1)
        self._threat_geom[ids] = torch.where(active.expand(n, K), geom, torch.full_like(geom, -1))
        self._threat_spawned[ids] = True

    def _threat_state(self):
        """Stacked (N, K, ...) world position, linear velocity, quaternion (w,x,y,z), body angular velocity of the threat bodies."""
        o = self._threat_objs
        return (torch.stack([t.data.root_pos_w for t in o], dim=1), torch.stack([t.data.root_lin_vel_w for t in o], dim=1),
                torch.stack([t.data.root_quat_w for t in o], dim=1), torch.stack([t.data.root_ang_vel_b for t in o], dim=1))

    def _sync_threats(self):
        """threat_pos_w / threat_vel_w mirror the simulated bodies (rewards, observations and the swept catch check read these)."""
        if self._threat_objs:
            self.threat_pos_w = torch.stack([t.data.root_pos_w for t in self._threat_objs], dim=1)
            self.threat_vel_w = torch.stack([t.data.root_lin_vel_w for t in self._threat_objs], dim=1)

    def _step_threats(self):
        """Once per policy step (from _pre_physics_step): decide the release, pick each threat's velocity command, turn it into an
        angle-style action with the shared LOS maths and set the rotor targets. The physics itself runs in _apply_threat_forces."""
        if self._K == 0:
            return
        if self._threat_active_k == 0:
            self._t_w_target[:] = self._t_hover_rpm
            return
        if self._threat_gains is None:
            raise RuntimeError("threat drones need set_threat_gains(gains_by_dist, max_tilt) before the first step")
        self._materialize_threats()
        self._sync_threats()
        N, K = self.num_envs, self._K
        if self._threat_spawn == "geometry":
            self._spawn_geometry()

        # handoff stage: the controller flies the drone first and the policy cannot dodge yet, so threats wait where they spawned;
        # after it each waits until the drone has closed to ITS release_frac of the distance it had left at the handoff (latched)
        handed = getattr(self, "_handed_off", None)
        flying = torch.ones(N, dtype=torch.bool, device=self.device) if handed is None else handed
        ref = self._start_dist if handed is None else getattr(self, "_handoff_h", self._start_dist)
        left = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=-1)
        reached = left.unsqueeze(-1) <= self._threat_release_frac_slot * ref.unsqueeze(-1)            # (N, K)
        self._threat_released = self._threat_released | (flying.unsqueeze(-1) & reached)
        moving = (self.threat_alive & self._threat_released).unsqueeze(-1)

        cap = self._threat_speed_cap
        if self._threat_policy is not None:      # custom (N,K,3) velocity command, still speed-limited
            v_chase = self._threat_policy(self)
            v_chase = v_chase * torch.clamp(cap / torch.linalg.norm(v_chase, dim=-1, keepdim=True).clamp_min(1e-6), max=1.0)
        else:                                    # limit-LOS: straight at the drone at the speed cap
            v_chase = chase_velocity(self._robot.data.root_pos_w.unsqueeze(1) - self.threat_pos_w, cap)
        v_des = torch.where(moving, v_chase, hold_velocity(self._threat_hold_w - self.threat_pos_w))

        _, vel, quat, ang_b = self._threat_state()
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat.reshape(-1, 4))
        g = self._threat_gains
        action, _, _ = velocity_to_action(
            v_des.reshape(-1, 3), vel.reshape(-1, 3), ang_b.reshape(-1, 3), roll, pitch, yaw, torch.zeros_like(yaw),
            hover_thrust=self._hover_thrust, kp_att=g["kp_att"], kd_att=g["kd_att"], kp_yaw=g["kp_yaw"], kd_yaw=g["kd_yaw"],
            max_tilt=g["max_tilt"], cruise_gain=g["cruise_gain"])
        desired = torch.clamp(action, self._t_low, self._t_high)
        desired[:, 0] = desired[:, 0] + self._hover_thrust
        self._t_w_target = tp.mixer_inversion(self._t_cfg, desired).view(N, K, 4)

    def _apply_threat_forces(self):
        """Every physics substep, the drone's rotor model for all N*K threat bodies (same functions as _apply_action)."""
        N, K = self.num_envs, self._K
        w = tp.motor_lag(self._t_rpm.reshape(-1, 4), self._t_w_target.reshape(-1, 4), self._t_cfg.motor_tau, PHYSICS_DT)
        self._t_rpm = w.view(N, K, 4)
        thrust = tp.net_combining_thrust(self._t_cfg, w)
        torque = tp.net_combining_torque(self._t_cfg, w).view(N, K, 3)
        _, vel, quat, _ = self._threat_state()
        vel, quat = vel.reshape(-1, 3), quat.reshape(-1, 4)
        drag_w = tp.drag_force(vel, self._t_cfg.drag_coeff, CROSS_SEC_AREA, AIR_DENS)
        wind_w = tp.wind_force(self._wind_vector.unsqueeze(1).expand(-1, K, -1).reshape(-1, 3), self._t_cfg.mass, K_WIND_COEFF)
        force_body = torch.zeros(N * K, 3, device=self.device)
        force_body[:, 2] = thrust
        force_body = (force_body + math_utils.quat_apply_inverse(quat, drag_w + wind_w)).view(N, K, 3)
        for k, obj in enumerate(self._threat_objs):
            obj.permanent_wrench_composer.set_forces_and_torques(forces=force_body[:, k].unsqueeze(1), torques=torque[:, k].unsqueeze(1))

    def nearest_threat_rel(self):
        """(N,3) vector threat - drone at the CLOSEST APPROACH during the last policy step, for the nearest alive threat;
        None when there are no threats. Swept, because two drones closing at 36 m/s move 0.6 m between 60 Hz steps and a
        point check at the end of the step would tunnel straight through a small catch radius."""
        if self._K == 0 or self._threat_active_k == 0:
            return None
        self._materialize_threats()
        self._sync_threats()
        r1 = self.threat_pos_w - self._robot.data.root_pos_w.unsqueeze(1)
        r0 = self._threat_prev_rel
        dr = r1 - r0
        t = (-(r0 * dr).sum(-1) / (dr * dr).sum(-1).clamp_min(1e-9)).clamp(0.0, 1.0)
        closest = r0 + t.unsqueeze(-1) * dr
        self._threat_prev_rel = r1.clone()
        d = torch.linalg.norm(closest, dim=-1)
        d = torch.where(self.threat_alive, d, torch.full_like(d, 1e9))
        idx = d.argmin(dim=1)
        rel = closest[torch.arange(self.num_envs, device=self.device), idx]
        any_alive = self.threat_alive.any(dim=1, keepdim=True)
        return torch.where(any_alive, rel, torch.full_like(rel, 1e9))

    def _threat_obs(self) -> torch.Tensor:
        """(N, THREAT_FEATURES*K): slots sorted nearest-first (dead last), dead slots all zero."""
        self._materialize_threats()
        self._sync_threats()
        rel = self.threat_pos_w - self._robot.data.root_pos_w.unsqueeze(1)
        rvel = self.threat_vel_w - self._robot.data.root_lin_vel_w.unsqueeze(1)
        key = torch.where(self.threat_alive, torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], 1e9))
        order = key.argsort(dim=1).unsqueeze(-1)
        rel, rvel = rel.gather(1, order.expand(-1, -1, 3)), rvel.gather(1, order.expand(-1, -1, 3))
        alive = self.threat_alive.gather(1, order.squeeze(-1)).unsqueeze(-1).float()
        feat = torch.cat([_symlog(rel, DIST_SCALE), rvel / VEL_SCALE, alive], dim=-1) * alive
        return feat.reshape(self.num_envs, -1)

    def _opponent_obs(self) -> torch.Tensor:
        """(N, OPPONENT_FEATURES*K) for obs_layout "v2": estimated position relative to the drone (symlog) + status, nearest-first, unknown
        slots all zero. The estimate comes from environmental/perception.py (a placeholder: the truth for now)."""
        if self._K == 0:
            return torch.zeros(self.num_envs, 0, device=self.device)
        self._materialize_threats()
        self._sync_threats()
        est, known, _source = estimate_opponents(self.threat_pos_w, self.threat_alive)
        rel = est - self._robot.data.root_pos_w.unsqueeze(1)
        key = torch.where(known, torch.linalg.norm(rel, dim=-1), torch.full_like(rel[..., 0], 1e9))
        order = key.argsort(dim=1).unsqueeze(-1)
        rel = rel.gather(1, order.expand(-1, -1, 3))
        status = known.gather(1, order.squeeze(-1)).unsqueeze(-1).float()
        return (torch.cat([_symlog(rel, DIST_SCALE), status], dim=-1) * status).reshape(self.num_envs, self._K * OPPONENT_FEATURES)

    def _ally_obs(self) -> torch.Tensor:
        """(N, ALLY_FEATURES*A) for obs_layout "v2": x, y, z relative to the drone (symlog), alive, id (slot index + 1) / A. A = cfg.max_allies,
        0 in 1v1 (zero width). No ally behaviour exists yet: positions stay where they were written."""
        rel = self.allied_pos_w - self._robot.data.root_pos_w.unsqueeze(1)
        a = self.allied_pos_w.shape[1]
        alive = self.allied_alive.unsqueeze(-1).float()
        ids = ((torch.arange(a, device=self.device) + 1.0) / max(a, 1)).view(1, a, 1).expand(self.num_envs, -1, -1)
        return (torch.cat([_symlog(rel, DIST_SCALE), alive, ids], dim=-1) * alive).reshape(self.num_envs, a * ALLY_FEATURES)

    # ---------------------------------------------------------------- threats end

    # ---------------------------------------------------------------- runner start
    # cfg.runner: the target is a drone (the learner's rotor model, flown by the shared LOS maths) that hovers on the target point until
    # the handoff, then flees straight away from the drone at its speed cap. _desired_pos_w follows its body once it is released, so the
    # observation, the rewards, the teacher and the hit check all see a moving target without knowing about it. The episode is lost
    # when the RUNNER leaves the sphere of radius escape_dist around its own spawn point (the target point it hovered on), wherever the
    # drone is (checked by the acro reward, see acro_reward_v1 "escape").

    def _sync_runner_body(self):
        """Same as _sync_threat_bodies: the runner's sphere gets the drone's mass and inertia."""
        if self._runner_obj is None:
            return
        idx = self._robot._ALL_INDICES.cpu()
        self._runner_obj.root_physx_view.set_masses(self._robot.root_physx_view.get_masses().clone(), idx)
        self._runner_obj.root_physx_view.set_inertias(self._robot.root_physx_view.get_inertias().clone(), idx)

    def set_runner_gains(self, gains_by_dist: dict, max_tilt: float, cruise_gain: float = CHASER_CRUISE_GAIN):
        """Attitude gains of the runner drone: the short-range entry of the PID table, like set_threat_gains. max_tilt in rad."""
        g = gains_by_dist[min(gains_by_dist, key=lambda key: float(key))]
        self._runner_gains = dict(kp_att=float(g["kp_att"]), kd_att=float(g["kd_att"]), kp_yaw=float(g["kp_yaw"]), kd_yaw=float(g["kd_yaw"]),
                                  max_tilt=float(max_tilt), cruise_gain=float(cruise_gain))

    def set_runner(self, max_speed: float, *, min_speed: float | None = None, escape_dist: float = 100.0, min_alt: float = 2.0):
        """Runner speed cap ~ U(min_speed, max_speed) per episode (m/s); the drone loses once the runner is more than escape_dist m from
        its own spawn point (sphere around where it started, not around the drone); the runner never flees below min_alt m."""
        if not self._runner:
            raise RuntimeError("set_runner needs cfg.runner = True (the runner body is created with the scene)")
        min_speed = max_speed if min_speed is None else min_speed
        if not 0.0 < min_speed <= max_speed:
            raise ValueError(f"need 0 < runner min_speed <= max_speed, got {min_speed} and {max_speed}")
        if escape_dist <= 0.0 or min_alt < 0.0:
            raise ValueError(f"need escape_dist > 0 and min_alt >= 0, got {escape_dist} and {min_alt}")
        self._runner_speed = (float(min_speed), float(max_speed))
        self._runner_escape_dist, self._runner_min_alt = float(escape_dist), float(min_alt)
        self._runner_fresh[:] = True

    def _materialize_runner(self):
        """Put the runner of every just-reset env on its target point, hovering. Lazy for the same reason as _materialize_threats:
        the handoff wrapper moves the target after _reset_idx."""
        if not bool(self._runner_fresh.any()):
            return
        ids = torch.nonzero(self._runner_fresh, as_tuple=False).squeeze(-1)
        n = ids.numel()
        pos = self._desired_pos_w[ids].clone()
        quat0 = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(n, 4)
        self._runner_obj.write_root_pose_to_sim(torch.cat([pos, quat0], dim=-1), ids)
        self._runner_obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=self.device), ids)
        self._r_rpm[ids] = self._t_hover_rpm
        self._r_w_target[ids] = self._t_hover_rpm
        self._runner_hold_w[ids] = pos
        lo, hi = self._runner_speed
        self._runner_cap[ids] = lo + (hi - lo) * torch.rand(n, 1, device=self.device)
        self._runner_released[ids] = False
        self._runner_vel_w[ids] = 0.0
        self._runner_fresh[ids] = False

    def _flee_velocity(self, runner_pos, drone_pos, cap, floor_z):
        """Straight away from the drone at the cap; the vertical part may not take it below floor + min_alt (it climbs back above)."""
        v = chase_velocity(runner_pos - drone_pos, cap)
        v_z_min = 2.0 * (floor_z + self._runner_min_alt - runner_pos[:, 2])     # 0 at min_alt, > 0 below it
        v[:, 2] = torch.maximum(v[:, 2], v_z_min)
        return v

    def _step_runner(self):
        """Once per policy step (from _pre_physics_step): release at the handoff, choose hold / flee, set the rotor targets."""
        if not self._runner:
            return
        if self._runner_gains is None:
            raise RuntimeError("the runner needs set_runner_gains(gains_by_dist, max_tilt) before the first step")
        self._materialize_runner()
        o = self._runner_obj
        pos, vel, quat, ang_b = o.data.root_pos_w, o.data.root_lin_vel_w.clone(), o.data.root_quat_w, o.data.root_ang_vel_b
        drone_p = self._robot.data.root_pos_w
        floor_z = self._terrain.env_origins[:, 2]
        handed = getattr(self, "_handed_off", None)
        flying = torch.ones_like(self._runner_released) if handed is None else handed
        launch = flying & ~self._runner_released
        if bool(launch.any()):
            # released already moving away at its cap (like the angle chasers), so a fast handoff does not get a free catch
            ids = torch.nonzero(launch, as_tuple=False).squeeze(-1)
            start = self._desired_pos_w[ids]                    # the hold point (the body may not be updated yet on a fresh reset)
            v0 = self._flee_velocity(start, drone_p[ids], self._runner_cap[ids], floor_z[ids])
            v0 = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True).clamp_min(1e-6) * self._runner_cap[ids]
            o.write_root_velocity_to_sim(torch.cat([v0, torch.zeros_like(v0)], dim=-1), ids)
            vel[ids] = v0
            self._runner_released[ids] = True
            self._runner_vel_w[ids] = v0
        flee = self._flee_velocity(pos, drone_p, self._runner_cap, floor_z)
        v_des = torch.where(self._runner_released.unsqueeze(-1), flee, hold_velocity(self._runner_hold_w - pos))
        roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat)
        g = self._runner_gains
        action, _, _ = velocity_to_action(
            v_des, vel, ang_b, roll, pitch, yaw, torch.zeros_like(yaw), hover_thrust=self._hover_thrust, kp_att=g["kp_att"],
            kd_att=g["kd_att"], kp_yaw=g["kp_yaw"], kd_yaw=g["kd_yaw"], max_tilt=g["max_tilt"], cruise_gain=g["cruise_gain"])
        desired = torch.clamp(action, self._t_low, self._t_high)
        desired[:, 0] = desired[:, 0] + self._hover_thrust
        self._r_w_target = tp.mixer_inversion(self._t_cfg, desired)

    def _apply_runner_forces(self):
        """Every physics substep: the drone's rotor model for the runner body (same functions as _apply_action)."""
        o = self._runner_obj
        self._r_rpm = tp.motor_lag(self._r_rpm, self._r_w_target, self._t_cfg.motor_tau, PHYSICS_DT)
        thrust = tp.net_combining_thrust(self._t_cfg, self._r_rpm)
        torque = tp.net_combining_torque(self._t_cfg, self._r_rpm)
        drag_w = tp.drag_force(o.data.root_lin_vel_w, self._t_cfg.drag_coeff, CROSS_SEC_AREA, AIR_DENS)
        wind_w = tp.wind_force(self._wind_vector, self._t_cfg.mass, K_WIND_COEFF)
        force_body = torch.zeros(self.num_envs, 3, device=self.device)
        force_body[:, 2] = thrust
        force_body = force_body + math_utils.quat_apply_inverse(o.data.root_quat_w, drag_w + wind_w)
        o.permanent_wrench_composer.set_forces_and_torques(forces=force_body.unsqueeze(1), torques=torque.unsqueeze(1))

    def _sync_runner(self):
        """After the physics: a released runner IS the target (position and velocity); a waiting one leaves the target where it is."""
        rel = self._runner_released.unsqueeze(-1)
        o = self._runner_obj
        self._desired_pos_w.copy_(torch.where(rel, o.data.root_pos_w, self._desired_pos_w))
        self._runner_vel_w.copy_(torch.where(rel, o.data.root_lin_vel_w, torch.zeros_like(self._runner_vel_w)))

    def runner_escaped(self) -> torch.Tensor | None:
        """(N,) bool: released runner outside the sphere of radius escape_dist around its spawn point (the drone lost); None without a
        runner. _runner_hold_w is that spawn point (the target point it hovered on until the policy took over)."""
        if not self._runner:
            return None
        return self._runner_released & (torch.linalg.norm(self._desired_pos_w - self._runner_hold_w, dim=-1) > self._runner_escape_dist)

    # ---------------------------------------------------------------- runner end


    def _pre_physics_step(self, actions: torch.Tensor):
        self._step_threats()
        self._step_runner()
        self._actions = torch.clamp(actions, self._action_low, self._action_high)
        real_action = self._actions.clone()
        real_action[:, 0] = real_action[:, 0] + self._hover_thrust
        if self._acro:
            # torques depend on the live body rate, so mixer inversion moves into every physics substep
            self._thrust_cmd = real_action[:, 0]
            self._rate_cmd = real_action[:, 1:4]
        else:
            self._w_target = tp.mixer_inversion(self._t_cfg, real_action)

    def _apply_action(self):
        if self._acro:
            torque_cmd = self._rate_ctrl.compute_torque(self._robot.data.root_ang_vel_b, self._rate_cmd, PHYSICS_DT)
            desired = torch.cat([self._thrust_cmd.unsqueeze(-1), torque_cmd], dim=-1)
            self._w_target = tp.mixer_inversion(self._t_cfg, desired)

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
        if self._K > 0:
            self._apply_threat_forces()
        if self._runner:
            self._apply_runner_forces()

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
        timer = zone_timer_features(dist.squeeze(-1), self._zone_count, self._zone_radius, self._zone_limit) if self._timer_obs else None
        # runner: target velocity relative to the drone, always the LAST inputs (appended, so smaller checkpoints still load)
        tail = [(self._runner_vel_w - vel) / VEL_SCALE] if self._runner else []
        if self._obs_v2:
            # base, timer, opponents, allies, flags -- fixed-capacity sections, see environmental/docs.md "Observation layouts"
            parts = [obs] + ([timer] if timer is not None else []) + [self._opponent_obs(), self._ally_obs(),
                                                                       mode_flags(self.num_envs, self.device)]
            return {"policy": torch.cat(parts + tail, dim=-1)}
        if self._K > 0:
            obs = torch.cat([obs, self._threat_obs()], dim=-1)
        if timer is not None:
            obs = torch.cat([obs, timer], dim=-1)
        return {"policy": torch.cat([obs] + tail, dim=-1)}

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
        self._prev_target_w.copy_(self._desired_pos_w)      # before the runner moves it: rewards compare against last step's target
        if self._runner:
            self._sync_runner()
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
        if self._K > 0:
            self.threat_alive[env_ids] = False       # re-spawned lazily, see _materialize_threats
            self._threat_spawned[env_ids] = False
            self._threat_geom[env_ids] = -1
            self._threat_fresh[env_ids] = True
        if self._runner:
            self._runner_fresh[env_ids] = True           # put back on the new target lazily, see _materialize_runner
            self._runner_released[env_ids] = False
            self._runner_vel_w[env_ids] = 0.0
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
        if self._acro:
            self._rate_ctrl.reset(env_ids)
            self._thrust_cmd[env_ids] = self._hover_thrust
            self._rate_cmd[env_ids] = 0.0

        self._start_dist[env_ids] = dist_mag
        self._prev_distance[env_ids] = dist_mag
        self._prev_position[env_ids] = spawn_local
        self._milestones_hit[env_ids] = False
        self._zone_count[env_ids] = 0
        self._zone_entered[env_ids] = False
        self._hit_done[env_ids] = False
        self._post_hit[env_ids] = 0

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

gym.register(
    id="Isaac-Acro-Drone-Direct-v0",
    entry_point=f"{__name__}:BaseDroneEnvIsaac",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}:AcroDroneEnvIsaacCfg"},
)
