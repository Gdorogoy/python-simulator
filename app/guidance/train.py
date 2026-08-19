import torch
import torch.nn as nn
from torch.distributions import Normal

device ="cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, action_low, action_high, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),

        )
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(hidden, 1)

        # action_low/action_high define the envs action_space box. The actor
        # head outputs unbounded numbers; we squash them through tanh and
        # rescale into this box (see scale_action / get_action_and_value)
        # instead of relying on env.step's np.clip to do it for us.
        # Registered as buffers so they're saved/loaded with the model.
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs):
        x = nn.functional.relu(self.shared(obs))
        action_mean = self.actor_mean(x)
        log_std=LOG_STD_MIN+0.5 *(LOG_STD_MAX-LOG_STD_MIN) *( torch.tanh(self.actor_log_std) +1) # was torch.clamp(self.actor_log_std, -3.0, 0.0) and before was none , changed because stopped ;earning in both of the cases
        action_std = torch.exp(log_std)
        state_value = self.critic_head(x).squeeze(-1)
        return action_mean, action_std, state_value

    def scale_action(self, raw_action):
        """Squash an unbounded raw action through tanh and rescale into
        [action_low, action_high]. Use this for deterministic (eval) actions."""
        squashed = torch.tanh(raw_action)
        return self.action_low + (squashed + 1) * 0.5 * (self.action_high - self.action_low)

    def get_action_and_value(self, obs, raw_action=None):
        """
        raw_action, if given, is the PRE-squash sample (what's stored in the
        rollout buffer) -- NOT the actual action sent to the env. We re-derive
        the squashed action deterministically from it so log_prob/entropy stay
        consistent between rollout collection and the PPO update.

        Returns (scaled_action, raw_action, log_prob, entropy, value).
        """
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
    n = len(rewards)
    advantages = torch.zeros(n, device=device)
    last_gae = 0.0

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
               global_timesteps_done: int = 0, global_total_timesteps: int = 0):
    """
    Runs the PPO epoch/minibatch loop. Returns a dict with the LAST executed
    epoch's averaged stats: policy_loss, value_loss, entropy_loss, approx_kl,
    grad_norm, early_stopped. Prints nothing except a single in-place
    progress line ("epoch(m/n) | step(m/n)") -- detailed metrics go to the
    caller to log (see phase_0_training.log_metrics).
    """
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
            # grad_norm is the pre-clip total norm returned by clip_grad_norm_,
            # captured before optimizer.step() applies the (possibly rescaled)
            # gradients.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
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

        # PPO early-stopping: if this epoch's updates moved the policy too far
        # from the data it was collected under, stop taking more epochs on
        # this batch rather than continuing to push off-distribution.
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
              ):
    """
    Runs PPO for `total_timesteps`.

    KEY CHANGE from before: accepts an EXISTING model/optimizer to resume
    training from, instead of always constructing a fresh one. Pass
    model=None (the default) to start fresh; pass a previously-returned
    model to continue training it -- this is what makes checkpointed,
    chunked training actually work instead of restarting from scratch
    every chunk.

    `ent_coef` is a caller-controlled entropy coefficient (default 0.01,
    matching the previous hardcoded value). Callers doing multi-chunk
    training should decay this over the course of the full run -- see
    phase_0_training.train() -- otherwise entropy keeps winning the loss
    tug-of-war indefinitely whenever the policy-gradient signal is weak.

    `global_timesteps_offset`/`global_total_timesteps` let a chunked caller
    (see phase_0_training.train()) report true whole-run progress in the
    "step(m/n)" progress line; they default to this call's own
    total_timesteps, preserving old behavior for callers that don't pass them.

    Now also returns `last_losses`, the dict from the final ppo_update call
    made in this chunk.
    """
    if model is None:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                             env.action_space.low, env.action_space.high).to(device)
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

            # store the PRE-squash raw action -- ppo_update recomputes the
            # squashed action/log_prob from this during the policy update.
            buffer.add(obs, raw_action.squeeze(0), log_prob.squeeze(0), reward, value.squeeze(0), done)

            episode_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)

                episode_end_states.append({               # <- NEW, captured BEFORE reset()
                    "position": env.drone_state.position,
                    "velocity": env.drone_state.velocity,
                    "orientation": env.drone_state.orientation,
                    "rotor_rpm": env.drone_state.rotor_rpm,
                    "final_dist": env.prev_distance,        # note: env.prev_distance, not self.prev_distance
                })




                episode_reward = 0.0


                obs, _ = env.reset()

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            last_value = model.get_action_and_value(last_obs_t)[4].item()

        advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, gamma, lam)
        timesteps_done += num_steps
        last_losses = ppo_update(model, optimizer, buffer, advantages, returns,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=ent_coef, num_epochs=10, batch_size=32,
                   global_timesteps_done=global_timesteps_offset + timesteps_done,
                   global_total_timesteps=global_total_timesteps, target_kl=target_kl)
                    #was  vf_coef=0.5


        with torch.no_grad():
            model.actor_log_std.clamp_(-2.0, -0.5)

    return model, optimizer, episode_rewards, last_losses


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