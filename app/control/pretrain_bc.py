import numpy as np
import torch
from torch.distributions import Normal

from app.guidance.train import ActorCritic, device
from app.reward_functions.reward_fn_phase1 import RewardFnPhase1


def pretrain_behavior_cloning(model, demo_path="app/control/demonstrations.npz",
                               obs=None, actions=None,
                               epochs=50, batch_size=256, lr=1e-3):
    """
    Trains on (obs, actions) arrays if given directly (e.g. DAgger's aggregated,
    growing dataset), otherwise loads them from demo_path.
    """
    if obs is None or actions is None:
        data = np.load(demo_path)
        obs = data["obs"]
        actions = data["actions"]

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(obs)

    for epoch in range(epochs):
        idx = torch.randperm(n)
        epoch_loss = 0.0
        epoch_std_sum = torch.zeros_like(model.actor_log_std)
        n_batches = 0

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            mean, std, _ = model.forward(obs[b])
            epoch_std_sum += std.detach().mean(dim=0)

            # actions[b] are real, bounded PID actions (Newtons/torque) -- map
            # them back into the SAME raw pre-tanh space `mean` lives in
            # (model.scale_action's inverse) before scoring, instead of
            # MSE-ing the raw mean directly against a bounded target it can
            # never exactly represent (tanh saturates). Then use the Gaussian
            # NLL (dist.log_prob), not MSE: squared error gets divided by
            # that action dimension's own learned variance, so a
            # large-magnitude dimension (e.g. thrust) doesn't automatically
            # dominate the loss over a small-magnitude one (e.g. yaw torque)
            # the way unweighted MSE would.
            half_range = 0.5 * (model.action_high - model.action_low)
            normalized = (actions[b] - model.action_low) / half_range - 1.0
            normalized = torch.clamp(normalized, -0.999, 0.999)  # keep atanh finite
            raw_target = torch.atanh(normalized)

            dist = Normal(mean, std)
            loss = -dist.log_prob(raw_target).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_std = (epoch_std_sum / n_batches).cpu().tolist()
        print(f"epoch {epoch+1}/{epochs}: avg_loss={epoch_loss / n_batches:.5f}  "
              f"avg_std(thrust,roll,pitch,yaw)={[f'{s:.3f}' for s in avg_std]}")

    return model


if __name__ == "__main__":
    from app.environmental.interceptor_drone import InterceptorDroneEnv

    reward_fn = RewardFnPhase1(
        hit_steps_streak=1500,
        phase1_pos_coef=0.25,
        hit_reward=5,
        oob_radius=300,  # matches collect_demonstrations.py's max distance (250m)
        hover_success_steps=None,
        streak_cap=60,
        outer_dist=1.0,
        inner_dist=0.3,
        hit_threshold=0.05,
        imitation_duration_steps=0,
        phase1_duration_steps=100,
    ).as_roadmap()
    env = InterceptorDroneEnv(reward_fn)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                        env.action_space.low,
                        env.action_space.high,

                        ).to(device)

    model = pretrain_behavior_cloning(model, demo_path="app/control/demonstrations.npz",
                                       epochs=55, batch_size=256, lr=1e-3)

    torch.save(model.state_dict(), "app/control/pretrained_bc.pt")
    print("saved pretrained weights to app/control/pretrained_bc.pt")