import numpy as np
import torch
import torch.nn as nn

from app.guidance.train import ActorCritic


def pretrain_behavior_cloning(model, demo_path="app/control/demonstrations.npz",
                               epochs=20, batch_size=256, lr=1e-3):
    data = np.load(demo_path)
    obs = torch.as_tensor(data["obs"], dtype=torch.float32)
    actions = torch.as_tensor(data["actions"], dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(obs)

    for epoch in range(epochs):
        idx = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]

            predicted_mean, _, _ = model.forward(obs[b])
            loss = nn.functional.mse_loss(predicted_mean, actions[b])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        print(f"epoch {epoch+1}/{epochs}: avg_loss={epoch_loss / n_batches:.5f}")

    return model


if __name__ == "__main__":
    from app.environmental.interceptor_drone import InterceptorDroneEnv
    from app.reward_functions.rewards import RewardConfig, make_reward_fn

    env = InterceptorDroneEnv(make_reward_fn(RewardConfig(oob_radius=7)))
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0],
                        env.action_space.low,
                        env.action_space.high,

                        )

    model = pretrain_behavior_cloning(model, demo_path="app/control/demonstrations.npz",
                                       epochs=20, batch_size=256, lr=1e-3)

    torch.save(model.state_dict(), "app/control/pretrained_bc.pt")
    print("saved pretrained weights to app/control/pretrained_bc.pt")