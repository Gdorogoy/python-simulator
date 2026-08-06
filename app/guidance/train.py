import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Linear(obs_dim, hidden)
        self.actor_mean = nn.Linear(hidden, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic_head = nn.Linear(hidden, 1)

    def forward(self, obs):
        x = nn.functional.relu(self.shared(obs))
        action_mean = self.actor_mean(x)
        log_std=LOG_STD_MIN+0.5 *(LOG_STD_MAX-LOG_STD_MIN) *( torch.tanh(self.actor_log_std) +1) # was torch.clamp(self.actor_log_std, -3.0, 0.0) and before was none , changed because stopped ;earning in both of the cases
        action_std = torch.exp(log_std)
        state_value = self.critic_head(x).squeeze(-1)
        return action_mean, action_std, state_value

    def get_action_and_value(self, obs, action=None):
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value


class RolloutBuffer:
    def __init__(self, num_steps: int, obs_dim: int, action_dim: int):
        self.obs = torch.zeros(num_steps, obs_dim)
        self.actions = torch.zeros(num_steps, action_dim)
        self.log_probs = torch.zeros(num_steps)
        self.rewards = torch.zeros(num_steps)
        self.values = torch.zeros(num_steps)
        self.dones = torch.zeros(num_steps)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = torch.as_tensor(obs)
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1


def compute_gae(rewards, values, dones, last_value, gamma: float, lam: float):
    n = len(rewards)
    advantages = torch.zeros(n)
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
               num_epochs: int, batch_size: int):
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    n = len(buffer.rewards)

    for epoch in range(num_epochs):
        idx = torch.randperm(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            _, new_log_probs, entropy, values = model.get_action_and_value(
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()


def ppo_train(env, total_timesteps: int, num_steps: int,
              gamma: float, lam: float, lr: float,
              model=None, optimizer=None):
    """
    Runs PPO for `total_timesteps`.

    KEY CHANGE from before: accepts an EXISTING model/optimizer to resume
    training from, instead of always constructing a fresh one. Pass
    model=None (the default) to start fresh; pass a previously-returned
    model to continue training it -- this is what makes checkpointed,
    chunked training actually work instead of restarting from scratch
    every chunk.
    """
    if model is None:
        model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    obs, _ = env.reset()
    episode_reward = 0.0
    timesteps_done = 0
    episode_rewards = []

    while timesteps_done < total_timesteps:
        buffer = RolloutBuffer(num_steps, env.observation_space.shape[0], env.action_space.shape[0])

        for step in range(num_steps):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _, value = model.get_action_and_value(obs_t)

            action_np = action.squeeze(0).numpy()
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated

            buffer.add(obs, action.squeeze(0), log_prob.squeeze(0), reward, value.squeeze(0), done)

            episode_reward += reward
            obs = next_obs
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0.0
                obs, _ = env.reset()

        with torch.no_grad():
            last_obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            last_value = model.get_action_and_value(last_obs_t)[3].item()

        advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, gamma, lam)
        ppo_update(model, optimizer, buffer, advantages, returns,
                   clip_eps=0.2, vf_coef=0.5, ent_coef=0.001, num_epochs=10, batch_size=64)
                    #ent_coef was 0.01

        timesteps_done += num_steps

    return model, optimizer, episode_rewards


def evaluate(model, env, n_episodes: int):
    n_success = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
            action = mean.squeeze(0).numpy()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward >= 15.0:
                n_success += 1
    print(f"Success rate: {n_success}/{n_episodes}")


if __name__ == "__main__":
    pass