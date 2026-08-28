"""Synchronous vectorized wrapper around N InterceptorDroneEnv instances, so a
policy's forward pass processes a batch of observations per step instead of one
at a time -- the only way a GPU does meaningful work here, since a single env's
per-step tensors are far too small on their own to justify the transfer."""
import numpy as np


class VecInterceptorDroneEnv:
    def __init__(self, envs: list):
        self.envs = envs
        self.num_envs = len(envs)
        self.observation_space = envs[0].observation_space
        self.action_space = envs[0].action_space
        self.max_steps = envs[0].max_steps
        self.dt = envs[0].dt

    def reset(self):
        obs = [env.reset()[0] for env in self.envs]
        return np.stack(obs).astype(np.float32)

    def step(self, actions):
        """actions: (num_envs, action_dim). Auto-resets any env that ends this step
        (standard vec-env convention), so a returned "done" row's obs already
        belongs to that env's next episode."""
        obs, rewards, terminated, truncated, infos = [], [], [], [], []
        for env, action in zip(self.envs, actions):
            o, r, term, trunc, info = env.step(action)
            if term or trunc:
                o, _ = env.reset()
            obs.append(o)
            rewards.append(r)
            terminated.append(term)
            truncated.append(trunc)
            infos.append(info)
        return (np.stack(obs).astype(np.float32), np.asarray(rewards, dtype=np.float32),
                np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool), infos)
