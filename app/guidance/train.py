import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_LOG_STD_MIN, DEFAULT_LOG_STD_MAX = -3.0, 0.0


def cosine_lr(base_lr, progress, min_ratio=0.01):
    """Cosine decay from base_lr down to base_lr*min_ratio as progress goes 0 -> 1."""
    factor = min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
    return base_lr * factor

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high,
                 hidden: int = 64, num_hidden_layers: int = 3, dropout: float = 0.0,
                 log_std_min: float = DEFAULT_LOG_STD_MIN, log_std_max: float = DEFAULT_LOG_STD_MAX):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # num_hidden_layers Linear layers, Tanh(+Dropout) between all but the
        # last -- matches the original fixed 3-layer trunk when left at defaults.
        layers = []
        in_dim = obs_dim
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden))
            if i < num_hidden_layers - 1:
                layers.append(nn.Tanh())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            in_dim = hidden
        self.shared = nn.Sequential(*layers)

        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(hidden, 1)

        # Actor head outputs unbounded values, squashed via tanh and rescaled
        # into [action_low, action_high] (see scale_action). Registered as
        # buffers so they're saved/loaded with the model.
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs):
        x = nn.functional.relu(self.shared(obs))
        action_mean = self.actor_mean(x)
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (torch.tanh(self.actor_log_std) + 1)
        action_std = torch.exp(log_std)
        state_value = self.critic_head(x).squeeze(-1)
        return action_mean, action_std, state_value

    def scale_action(self, raw_action):
        """Squash an unbounded raw action through tanh and rescale into
        [action_low, action_high]. Use this for deterministic (eval) actions."""
        squashed = torch.tanh(raw_action)
        return self.action_low + (squashed + 1) * 0.5 * (self.action_high - self.action_low)

    def get_action_and_value(self, obs, raw_action=None):
        """If raw_action is given (the pre-squash rollout-buffer sample), re-derive
        the squashed action from it so log_prob/entropy match rollout collection.
        Returns (scaled_action, raw_action, log_prob, entropy, value)."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)

        if raw_action is None:
            raw_action = dist.sample()

        squashed = torch.tanh(raw_action)
        half_range = 0.5 * (self.action_high - self.action_low)
        scaled_action = self.action_low + (squashed + 1) * half_range

        # change-of-variables correction for the tanh + affine rescale, so
        # log_prob is the density of the action actually sent to the env,
        # not of the pre-squash Gaussian sample.
        log_prob = dist.log_prob(raw_action).sum(-1) - torch.log(half_range * (1 - squashed.pow(2)) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)
        return scaled_action, raw_action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, num_steps: int, obs_dim: int, action_dim: int):
        self.obs = torch.zeros(num_steps, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, action_dim, device=device)
        self.log_probs = torch.zeros(num_steps, device=device)
        self.rewards = torch.zeros(num_steps, device=device)
        self.values = torch.zeros(num_steps, device=device)
        self.dones = torch.zeros(num_steps, device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs, device=device)
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    """Also handles the vectorized-envs case: pass rewards/values/dones shaped
    (num_steps, num_envs) and last_value as a (num_envs,) tensor -- the backward
    recursion below then runs across all envs at once via broadcasting."""
    n = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(last_value) if torch.is_tensor(last_value) else 0.0

    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, buffer, advantages, returns,
               clip_eps: float, vf_coef: float, ent_coef: float,
               num_epochs: int, batch_size: int, target_kl: float = 0.02,
               max_grad_norm: float = 0.5,
               global_timesteps_done: int = 0, global_total_timesteps: int = 0):
    """Runs the PPO epoch/minibatch loop. Returns a dict with the last executed
    epoch's averaged stats (policy_loss, value_loss, entropy_loss, approx_kl,
    grad_norm, early_stopped)."""
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = len(buffer.rewards)

    early_stopped = False
    last_epoch_stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0,
                         "approx_kl": 0.0, "grad_norm": 0.0}

    for epoch in range(num_epochs):
        idx = torch.randperm(n)
        epoch_policy, epoch_value, epoch_entropy, epoch_grad, epoch_kl = [], [], [], [], []

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            _, _, new_log_probs, entropy, values = model.get_action_and_value(
                buffer.obs[b], buffer.actions[b])

            ratio = torch.exp(new_log_probs - buffer.log_probs[b])
            unclipped = ratio * advantages[b]
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages[b]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_loss = ((values - returns[b]) ** 2).mean()
            entropy_loss = -entropy.mean()
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            # grad_norm is the pre-clip total norm, captured before step() applies it.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (buffer.log_probs[b] - new_log_probs).mean().item()

            epoch_policy.append(policy_loss.item())
            epoch_value.append(value_loss.item())
            epoch_entropy.append(entropy_loss.item())
            epoch_grad.append(grad_norm.item())
            epoch_kl.append(approx_kl)

        print(f"\repoch({epoch + 1}/{num_epochs}) | step({global_timesteps_done}/{global_total_timesteps})",
              end="", flush=True)

        last_epoch_stats = {
            "policy_loss": sum(epoch_policy) / len(epoch_policy),
            "value_loss": sum(epoch_value) / len(epoch_value),
            "entropy_loss": sum(epoch_entropy) / len(epoch_entropy),
            "approx_kl": sum(epoch_kl) / len(epoch_kl),
            "grad_norm": sum(epoch_grad) / len(epoch_grad),
        }

        # Early-stop epochs on this batch once the policy has drifted too far from it.
        if last_epoch_stats["approx_kl"] > target_kl:
            early_stopped = True
            break

    last_epoch_stats["early_stopped"] = early_stopped
    return last_epoch_stats


def ppo_train(env, total_timesteps: int, num_steps: int,
              gamma: float, lam: float, lr: float,
              target_kl: float,
              model=None, optimizer=None, ent_coef: float = 0.015,
              global_timesteps_offset: int = 0, global_total_timesteps: int = None,
              num_epochs: int = 10, batch_size: int = 32,
              clip_eps: float = 0.2, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
              hidden: int = 64, num_hidden_layers: int = 3, dropout: float = 0.0,
              log_std_min: float = DEFAULT_LOG_STD_MIN, log_std_max: float = DEFAULT_LOG_STD_MAX,
              log_std_clamp_min: float = -3.0, log_std_clamp_max: float = -0.5,
              ):
    """Runs PPO for `total_timesteps`. Pass model=None (default) to start fresh,
    or a previously-returned model/optimizer to resume chunked training.
    Returns (model, optimizer, episode_rewards, last_losses)."""
    if model is None:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                             env.action_space.low, env.action_space.high,
                             hidden=hidden, num_hidden_layers=num_hidden_layers, dropout=dropout,
                             log_std_min=log_std_min, log_std_max=log_std_max).to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if global_total_timesteps is None:
        global_total_timesteps = total_timesteps

    obs, _ = env.reset()
    episode_reward = 0.0
    timesteps_done = 0
    episode_rewards = []
    episode_end_states = []
    last_losses = {}

    while timesteps_done < total_timesteps:
        buffer = RolloutBuffer(num_steps, env.observation_space.shape[0], env.action_space.shape[0])

        for step in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, raw_action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            if truncated and not terminated:
                # Time-limit cutoff, not a real terminal state: bootstrap the missing
                # future value into this step's reward instead of treating it as zero.
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    bootstrap_value = model.forward(next_obs_t)[2].item()
                reward = reward + gamma * bootstrap_value

            # Store the pre-squash raw action; ppo_update recomputes the squashed
            # action/log_prob from it during the policy update.
            buffer.add(obs, raw_action.squeeze(0), log_prob.squeeze(0), reward, value.squeeze(0), done)

            episode_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)

                # Captured before env.reset() overwrites drone_state.
                episode_end_states.append({
                    "position": env.drone_state.position,
                    "velocity": env.drone_state.velocity,
                    "orientation": env.drone_state.orientation,
                    "rotor_rpm": env.drone_state.rotor_rpm,
                    "final_dist": env.prev_distance,
                })
                episode_reward = 0.0


                obs, _ = env.reset()

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model.get_action_and_value(last_obs_t)[4].item()

        advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, gamma, lam)
        timesteps_done += num_steps
        last_losses = ppo_update(model, optimizer, buffer, advantages, returns,
                   clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef, num_epochs=num_epochs,
                   batch_size=batch_size, max_grad_norm=max_grad_norm,
                   global_timesteps_done=global_timesteps_offset + timesteps_done,
                   global_total_timesteps=global_total_timesteps, target_kl=target_kl)

        with torch.no_grad():
            # Keeps action noise from collapsing to (near-)zero or exploding.
            model.actor_log_std.clamp_(log_std_clamp_min, log_std_clamp_max)

    return model, optimizer, episode_rewards, last_losses


def vec_ppo_train(vec_env, total_timesteps: int, num_steps: int,
                   gamma: float, lam: float, lr: float,
                   target_kl: float,
                   model=None, optimizer=None, ent_coef: float = 0.015,
                   global_timesteps_offset: int = 0, global_total_timesteps: int = None,
                   num_epochs: int = 10, batch_size: int = 32,
                   clip_eps: float = 0.2, vf_coef: float = 0.5, max_grad_norm: float = 0.5,
                   hidden: int = 64, num_hidden_layers: int = 3, dropout: float = 0.0,
                   log_std_min: float = DEFAULT_LOG_STD_MIN, log_std_max: float = DEFAULT_LOG_STD_MAX,
                   log_std_clamp_min: float = -3.0, log_std_clamp_max: float = -0.5,
                   ):
    """Same as ppo_train, but collects rollouts across vec_env.num_envs environments
    in lockstep each step -- one batched (num_envs, obs_dim) forward pass instead of
    num_envs separate ones, which is what actually gives a GPU something to chew on.
    vec_env is a VecInterceptorDroneEnv. Returns (model, optimizer, episode_rewards,
    last_losses), same as ppo_train (no per-episode drone-state snapshots here --
    nothing downstream of the vectorized path currently consumes those)."""
    num_envs = vec_env.num_envs
    obs_dim = vec_env.observation_space.shape[0]
    action_dim = vec_env.action_space.shape[0]

    if model is None:
        model = ActorCritic(obs_dim, action_dim, vec_env.action_space.low, vec_env.action_space.high,
                             hidden=hidden, num_hidden_layers=num_hidden_layers, dropout=dropout,
                             log_std_min=log_std_min, log_std_max=log_std_max).to(device)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if global_total_timesteps is None:
        global_total_timesteps = total_timesteps

    obs = vec_env.reset()
    running_episode_reward = np.zeros(num_envs, dtype=np.float32)
    timesteps_done = 0
    episode_rewards = []
    last_losses = {}

    while timesteps_done < total_timesteps:
        buf_obs = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        buf_actions = torch.zeros(num_steps, num_envs, action_dim, device=device)
        buf_log_probs = torch.zeros(num_steps, num_envs, device=device)
        buf_rewards = torch.zeros(num_steps, num_envs, device=device)
        buf_values = torch.zeros(num_steps, num_envs, device=device)
        buf_dones = torch.zeros(num_steps, num_envs, device=device)

        for step in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                action, raw_action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = vec_env.step(action_np)
            done = terminated | truncated

            # Same time-limit bootstrap as ppo_train, applied only to the envs that
            # actually truncated (not terminated) this step.
            trunc_only = truncated & ~terminated
            if trunc_only.any():
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                    bootstrap_values = model.forward(next_obs_t)[2].cpu().numpy()
                reward = reward + gamma * bootstrap_values * trunc_only.astype(np.float32)

            buf_obs[step] = obs_t
            buf_actions[step] = raw_action
            buf_log_probs[step] = log_prob
            buf_rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            buf_values[step] = value
            buf_dones[step] = torch.as_tensor(done, dtype=torch.float32, device=device)

            running_episode_reward += reward
            for i in np.flatnonzero(done):
                episode_rewards.append(float(running_episode_reward[i]))
                running_episode_reward[i] = 0.0

            obs = next_obs

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            last_value = model.get_action_and_value(last_obs_t)[4]

        advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, last_value, gamma, lam)
        timesteps_done += num_steps * num_envs

        # Flatten (num_steps, num_envs, ...) -> (num_steps*num_envs, ...); ppo_update
        # only reads these five attributes off `buffer`, so a plain namespace works.
        flat_buffer = SimpleNamespace(
            obs=buf_obs.reshape(-1, obs_dim), actions=buf_actions.reshape(-1, action_dim),
            log_probs=buf_log_probs.reshape(-1), rewards=buf_rewards.reshape(-1),
            values=buf_values.reshape(-1), dones=buf_dones.reshape(-1),
        )

        last_losses = ppo_update(model, optimizer, flat_buffer, advantages.reshape(-1), returns.reshape(-1),
                   clip_eps=clip_eps, vf_coef=vf_coef, ent_coef=ent_coef, num_epochs=num_epochs,
                   batch_size=batch_size, max_grad_norm=max_grad_norm,
                   global_timesteps_done=global_timesteps_offset + timesteps_done,
                   global_total_timesteps=global_total_timesteps, target_kl=target_kl)

        with torch.no_grad():
            model.actor_log_std.clamp_(log_std_clamp_min, log_std_clamp_max)

    return model, optimizer, episode_rewards, last_losses


def warmup_critic(env, model, num_rounds: int, num_steps: int, gamma: float, lam: float, lr: float = 3e-5,
                   max_grad_norm: float = 0.5):
    """Collects rollouts with the current (untouched) actor and updates only
    critic_head for num_rounds, so a BC/DAgger-loaded critic isn't still at
    random init when real PPO updates start. lr is cosine-decayed across
    rounds since each round's regression target is non-stationary."""
    critic_optimizer = torch.optim.Adam(model.critic_head.parameters(), lr=lr)
    obs, _ = env.reset()

    for round_idx in range(num_rounds):
        round_lr = cosine_lr(lr, round_idx / max(1, num_rounds - 1))
        for param_group in critic_optimizer.param_groups:
            param_group['lr'] = round_lr

        buffer = RolloutBuffer(num_steps, env.observation_space.shape[0], env.action_space.shape[0])

        for step in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, raw_action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.squeeze(0).cpu().numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            if truncated and not terminated:
                with torch.no_grad():
                    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                    bootstrap_value = model.forward(next_obs_t)[2].item()
                reward = reward + gamma * bootstrap_value

            buffer.add(obs, raw_action.squeeze(0), log_prob.squeeze(0), reward, value.squeeze(0), done)

            obs = next_obs
            if done:
                obs, _ = env.reset()

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model.get_action_and_value(last_obs_t)[4].item()

        _, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, gamma, lam)

        value_loss = None
        for _ in range(4):
            _, _, _, _, values = model.get_action_and_value(buffer.obs, buffer.actions)
            value_loss = ((values - returns) ** 2).mean()

            model.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic_head.parameters(), max_norm=max_grad_norm)
            critic_optimizer.step()

        print(f"[critic warmup] round {round_idx + 1}/{num_rounds}  lr={round_lr:.2e}  value_loss={value_loss.item():.4f}")

    return model


def evaluate(model, env, n_episodes: int):
    n_success = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
                action = model.scale_action(mean)
            action = action.squeeze(0).cpu().numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward >= 15.0:
                n_success += 1
    print(f"Success rate: {n_success}/{n_episodes}")


if __name__ == "__main__":
    pass