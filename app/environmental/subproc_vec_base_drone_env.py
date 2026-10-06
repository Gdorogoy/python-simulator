"""Multiprocess (spawn) vec-env of numpy BaseDroneEnvs, used for CPU-side evaluation. See docs.md "SubprocVecBaseDroneEnv"."""
import json
import multiprocessing as mp
import os

import numpy as np

_CTX = mp.get_context("spawn")

PID_GAINS_BY_DIST_PATH = "app/control/best_pid_gains_per_dist.json"


def _select_pid_teacher(env, gains_by_dist):
    """Swap env.pid_teacher to the per-distance gain set matching this episode's start distance."""
    from app.control.pid import PIDController

    keys = list(gains_by_dist.keys())
    nearest = min(keys, key=lambda k: abs(float(k) - env.prev_distance))
    env.pid_teacher = PIDController(**gains_by_dist[nearest])


def _worker_main(reward_fn_factory, factory_kwargs, p, target_pairs, n_envs, conn, max_steps):
    from app.environmental.base_drone_env import BaseDroneEnv

    with open(PID_GAINS_BY_DIST_PATH) as f:
        gains_by_dist = json.load(f)

    reward_fn = reward_fn_factory(p, **factory_kwargs)
    envs = [BaseDroneEnv(reward_fn, target_pairs=target_pairs, max_steps=max_steps) for _ in range(n_envs)]

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "reset":
                obs = []
                for env in envs:
                    o, _ = env.reset()
                    _select_pid_teacher(env, gains_by_dist)
                    obs.append(o)
                conn.send(np.stack(obs).astype(np.float32))
            elif cmd == "step":
                obs, rewards, terminated, truncated, infos = [], [], [], [], []
                for env, action in zip(envs, payload):
                    o, r, term, trunc, info = env.step(action)
                    if term or trunc:
                        # keep the real terminal obs before auto-reset (needed to bootstrap truncations)
                        info = {**info, "terminal_observation": o}
                        o, _ = env.reset()
                        _select_pid_teacher(env, gains_by_dist)
                    obs.append(o)
                    rewards.append(r)
                    terminated.append(term)
                    truncated.append(trunc)
                    infos.append(info)
                conn.send((
                    np.stack(obs).astype(np.float32),
                    np.asarray(rewards, dtype=np.float32),
                    np.asarray(terminated, dtype=bool),
                    np.asarray(truncated, dtype=bool),
                    infos,
                ))
            elif cmd == "step_teach":
                # "step" + the PID action for the PRE-step state (imitation stage BC pairs)
                obs, rewards, terminated, truncated, infos, pid_actions = [], [], [], [], [], []
                for env, action in zip(envs, payload):
                    pid_actions.append(env.pid_teacher.compute_action(
                        env.drone_state, env.target_pos, env.target_yaw))
                    o, r, term, trunc, info = env.step(action)
                    if term or trunc:
                        info = {**info, "terminal_observation": o}
                        o, _ = env.reset()
                        _select_pid_teacher(env, gains_by_dist)
                    obs.append(o)
                    rewards.append(r)
                    terminated.append(term)
                    truncated.append(trunc)
                    infos.append(info)
                conn.send((
                    np.stack(obs).astype(np.float32),
                    np.asarray(rewards, dtype=np.float32),
                    np.asarray(terminated, dtype=bool),
                    np.asarray(truncated, dtype=bool),
                    infos,
                    np.stack(pid_actions).astype(np.float32),
                ))
            elif cmd == "get_target_positions":
                # target_pos, start_dist (fixed at reset) and live_dist (current distance)
                targets = np.stack([env.target_pos for env in envs]).astype(np.float32)
                start_dists = np.array([env.start_dist for env in envs], dtype=np.float32)
                live_dists = np.array([env.prev_distance for env in envs], dtype=np.float32)
                conn.send((targets, start_dists, live_dists))
            elif cmd == "close":
                conn.close()
                return
    except EOFError:
        return


class SubprocVecBaseDroneEnv:
    """Vec-env API (reset/step/spaces) over worker processes; call close() explicitly (e.g. in finally)."""

    def __init__(self, reward_fn_factory, p: dict, target_pairs, num_envs: int, num_workers: int = None,
                 factory_kwargs: dict = None, max_steps: int = 15_000):
        from app.environmental.base_drone_env import BaseDroneEnv

        factory_kwargs = factory_kwargs or {}

        if num_workers is None:
            # leave one core for the main (GPU) process
            num_workers = max(1, (os.cpu_count() or 2) - 1)
        num_workers = max(1, min(num_workers, num_envs))

        base, rem = divmod(num_envs, num_workers)
        self._counts = [base + (1 if i < rem else 0) for i in range(num_workers)]
        self._counts = [c for c in self._counts if c > 0]
        # envs per worker, in the order per-env arrays are concatenated
        self.counts = list(self._counts)

        self.num_envs = num_envs

        # local throwaway env just to read spaces / max_steps / dt
        probe_env = BaseDroneEnv(reward_fn_factory(p, **factory_kwargs), target_pairs=target_pairs,
                                  max_steps=max_steps)
        self.observation_space = probe_env.observation_space
        self.action_space = probe_env.action_space
        self.max_steps = probe_env.max_steps
        self.dt = probe_env.dt
        del probe_env

        self._procs = []
        self._conns = []
        for n_envs_i in self._counts:
            parent_conn, child_conn = _CTX.Pipe()
            proc = _CTX.Process(target=_worker_main, args=(reward_fn_factory, factory_kwargs, p, target_pairs,
                                                             n_envs_i, child_conn, max_steps), daemon=True)
            proc.start()
            child_conn.close()  # only the worker's copy of this end should stay open
            self._procs.append(proc)
            self._conns.append(parent_conn)

    def reset(self):
        for conn in self._conns:
            conn.send(("reset", None))
        parts = [conn.recv() for conn in self._conns]
        return np.concatenate(parts, axis=0)

    def step(self, actions):
        offset = 0
        for conn, n_envs_i in zip(self._conns, self._counts):
            conn.send(("step", actions[offset:offset + n_envs_i]))
            offset += n_envs_i
        results = [conn.recv() for conn in self._conns]
        obs = np.concatenate([r[0] for r in results], axis=0)
        rewards = np.concatenate([r[1] for r in results], axis=0)
        terminated = np.concatenate([r[2] for r in results], axis=0)
        truncated = np.concatenate([r[3] for r in results], axis=0)
        infos = [info for r in results for info in r[4]]
        return obs, rewards, terminated, truncated, infos

    def step_with_pid_actions(self, actions):
        """step() plus each env's PID action for the pre-step state (imitation stage only)."""
        offset = 0
        for conn, n_envs_i in zip(self._conns, self._counts):
            conn.send(("step_teach", actions[offset:offset + n_envs_i]))
            offset += n_envs_i
        results = [conn.recv() for conn in self._conns]
        obs = np.concatenate([r[0] for r in results], axis=0)
        rewards = np.concatenate([r[1] for r in results], axis=0)
        terminated = np.concatenate([r[2] for r in results], axis=0)
        truncated = np.concatenate([r[3] for r in results], axis=0)
        infos = [info for r in results for info in r[4]]
        pid_actions = np.concatenate([r[5] for r in results], axis=0)
        return obs, rewards, terminated, truncated, infos, pid_actions

    def get_target_positions(self):
        """Live (target_pos, start_dist, live_dist) of every env's current episode."""
        for conn in self._conns:
            conn.send(("get_target_positions", None))
        results = [conn.recv() for conn in self._conns]
        targets = np.concatenate([r[0] for r in results], axis=0)
        start_dists = np.concatenate([r[1] for r in results], axis=0)
        live_dists = np.concatenate([r[2] for r in results], axis=0)
        return targets, start_dists, live_dists

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for conn in self._conns:
            conn.close()
