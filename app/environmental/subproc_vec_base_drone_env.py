"""Multiprocess vec-env: shards N BaseDroneEnv instances across worker
processes (one core each) so physics stepping actually uses more than one CPU
core. VecBaseDroneEnv's Python for-loop is single-threaded and is the
measured throughput bottleneck (~92 per-env steps/sec regardless of network
architecture size -- GPU/NN cost is negligible by comparison).

Uses spawn, not fork: the main process already initializes CUDA (model lives
on the GPU) by the time a vec-env gets built, and forking a process after CUDA
init is unsafe (driver-internal state/locks aren't fork-safe). spawn starts
each worker as a fresh interpreter instead, at the cost of everything crossing
the process boundary needing to be picklable.

Reward-fn closures from chain_reward_fns are NOT picklable (nested functions),
so a constructed env can never cross that boundary -- each worker rebuilds its
own envs locally from picklable plain data (a plain top-level reward_fn_factory
function reference, a param dict `p`, and a target_pairs list of numpy arrays)
instead of receiving env/reward-fn objects. Reward-fn-agnostic: pass
build_phase1_reward_fn, build_base_reward_fn (both in app.reward_functions.rewards),
or any other plain module-level factory with the same (p, **kwargs) -> reward_fn
signature.
"""
import json
import multiprocessing as mp
import os

import numpy as np

_CTX = mp.get_context("spawn")

PID_GAINS_BY_DIST_PATH = "app/control/best_pid_gains_per_dist.json"


def _select_pid_teacher(env, gains_by_dist):
    """Swaps env.pid_teacher to the gain set for whichever distance this
    episode's target actually landed at (env.prev_distance, set by reset()) --
    BaseDroneEnv's own default pid_teacher loads ONE generic gain set
    (app/control/best_pid_gains.json) regardless of target distance, which is
    a much worse imitation teacher than the per-distance gains dagger.py/
    collect_demonstrations.py were trained with (e.g. kp_pos=10.98 at 3m vs
    3.09 at 50m -- a single fixed gain can't be right for both)."""
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
                        # Stashed before the auto-reset overwrites o -- callers
                        # bootstrapping a truncated episode's value need the true
                        # terminal state, not the next episode's fresh-spawn obs.
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
                # Same as "step", but also returns each env's PID-teacher action
                # for the state that PRODUCED `payload`'s actions (computed before
                # stepping, so it's paired with the obs the caller already has --
                # not the next state). Used only by the imitation-learning stage
                # (base_training.py) to build (obs, pid_action) BC pairs from an
                # on-policy rollout; ignored by plain PPO rollout collection.
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
                # (num_envs_here, 3) target_pos + start_dist (fixed at reset --
                # "what is training actually spawning targets at right now") +
                # live_dist (env.prev_distance, mutated every reward_func call --
                # "how far is this env's drone from its target RIGHT NOW", which
                # diverges from start_dist once a policy starts drifting/failing
                # mid-episode). Both are meaningful and distinct, so both ship.
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
    """Same interface as VecBaseDroneEnv (reset/step/num_envs/
    observation_space/action_space/max_steps/dt), plus close() to shut the
    worker processes down -- callers must call it explicitly (e.g. in a
    finally:) since garbage collection alone won't clean up child processes.

    p and target_pairs must be plain picklable data; envs are built fresh
    inside each worker, never pickled across the boundary.

    reward_fn_factory must be a plain module-level function (not a lambda or
    closure -- those aren't picklable across the spawn boundary), called as
    reward_fn_factory(p, **factory_kwargs) to build each worker's reward_fn.
    See phase_1_training.build_phase1_reward_fn / app.reward_functions.rewards.build_base_reward_fn."""

    def __init__(self, reward_fn_factory, p: dict, target_pairs, num_envs: int, num_workers: int = None,
                 factory_kwargs: dict = None, max_steps: int = 15_000):
        from app.environmental.base_drone_env import BaseDroneEnv

        factory_kwargs = factory_kwargs or {}

        if num_workers is None:
            # Leave the main process a core of its own for the GPU-side loop
            # and pipe I/O -- oversubscribing every core doesn't help throughput.
            num_workers = max(1, (os.cpu_count() or 2) - 1)
        num_workers = max(1, min(num_workers, num_envs))

        base, rem = divmod(num_envs, num_workers)
        self._counts = [base + (1 if i < rem else 0) for i in range(num_workers)]
        self._counts = [c for c in self._counts if c > 0]
        # Public copy: how many envs each worker owns, in the same order the
        # per-env arrays from step()/get_target_positions() are concatenated in --
        # callers that want a per-worker breakdown (e.g. plot_worker_metrics)
        # split those flat (num_envs,) arrays using this.
        self.counts = list(self._counts)

        self.num_envs = num_envs

        # Probe obs/action space + max_steps/dt from one throwaway local env --
        # cheap, avoids a round-trip to a worker just to read static attributes.
        # Never crosses a process boundary, so its reward-fn closure is fine here.
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
        """Like step(), but also returns (num_envs, action_dim) pid_actions --
        each env's PID-teacher action for the SAME pre-step state as the obs
        each env had when `actions` was computed (not the next state).
        Only for the imitation-learning stage; plain PPO rollout uses step()."""
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
        """(num_envs, 3) target_pos + (num_envs,) start_dist (fixed at that
        episode's reset) + (num_envs,) live_dist (current distance-to-target,
        mutated every reward_func call -- diverges from start_dist once a
        policy drifts/fails mid-episode), for every env's CURRENT episode --
        a live snapshot of what targets training is actually spawning right
        now, not just the static target_pairs list."""
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
