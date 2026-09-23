import numpy as np
import torch
from torch.distributions import Normal

from app.guidance.train import ActorCritic, device


def pretrain_behavior_cloning(model, demo_path="app/control/demonstrations.npz",
                               obs=None, actions=None, weights=None,
                               epochs=50, batch_size=256, lr=1e-3, weight_decay=1e-4):
    """
    Trains on (obs, actions) arrays if given directly (e.g. DAgger's aggregated,
    growing dataset), otherwise loads them from demo_path.

    `weights`, if given, is a per-sample array (same length as obs/actions) used to
    weight each sample's contribution to the loss -- e.g. DAgger's recency weights,
    so older rounds fade instead of counting equally with fresh corrections.
    Unweighted (None) reproduces plain uniform BC.
    """
    if obs is None or actions is None:
        data = np.load(demo_path)
        obs = data["obs"]
        actions = data["actions"]

    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
    weights = torch.as_tensor(weights, dtype=torch.float32, device=device) if weights is not None else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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

            # Map bounded PID actions into raw pre-tanh space (model.scale_action's inverse) and
            # score with Gaussian NLL instead of MSE, so large-magnitude dims don't dominate small ones.
            half_range = 0.5 * (model.action_high - model.action_low)
            normalized = (actions[b] - model.action_low) / half_range - 1.0
            normalized = torch.clamp(normalized, -0.999, 0.999)  # keep atanh finite
            raw_target = torch.atanh(normalized)

            dist = Normal(mean, std)
            per_sample_nll = -dist.log_prob(raw_target).mean(dim=-1)  # per-sample so weights can apply before collapsing
            if weights is not None:
                w = weights[b]
                loss = (per_sample_nll * w).sum() / w.sum()
            else:
                loss = per_sample_nll.mean()

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
    # Needs neither isaaclab nor a GPU -- BaseDroneEnv below only supplies obs/action shape.
    import argparse

    from app.environmental.base_drone_env import BaseDroneEnv
    from app.reward_functions.rewards import reward_func

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-path", dest="demo_path", default="app/control/demonstrations_omni.npz")
    parser.add_argument("--out-path", dest="out_path", default="app/control/pretrained_bc.pt")
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--num-hidden-layers", type=int, default=4)
    args = parser.parse_args()

    env = BaseDroneEnv(reward_func)  # only used for observation_space/action_space shape

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                        env.action_space.low,
                        env.action_space.high,
                        hidden=args.hidden, num_hidden_layers=args.num_hidden_layers,
                        ).to(device)

    model = pretrain_behavior_cloning(model, demo_path=args.demo_path,
                                       epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)

    torch.save(model.state_dict(), args.out_path)
    print(f"saved pretrained weights to {args.out_path}")