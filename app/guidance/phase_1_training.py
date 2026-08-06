"""
Phase 1 training: basic navigation toward a constant static target.


### no relevatnt now its [1,1,1]
Object placed 5m from the drone at 45 deg, with a +1m y-offset baked in
(to also teach some off-axis correction, not just a pure planar approach).

"""
from datetime import datetime

import numpy as np
import torch
import os


from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic, ppo_train, evaluate

import logging

os.makedirs("logs", exist_ok=True)
log_filename = f"logs/phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        logging.FileHandler(log_filename), # TO WRITE
        logging.StreamHandler(),  # TO LOG INTO CONSOLE
    ],
)
log = logging.getLogger(__name__)

log.info(f"Logging this run to: {log_filename}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class TrainConfig:
    TARGET_DISTANCE = 1
    TARGET_ANGLE_DEG = 45.0
    TARGET_Y_OFFSET = 1.0

    TOTAL_TIMESTEPS = 2_000_000
    NUM_STEPS = 5000                   # rollout length per PPO update
    GAMMA = 0.97
    LAM = 0.95
    LR = 3e-4 # was originaly 3e-5 , but drone is not doing anything so now th gradient will be 100 times bigger

    CHECKPOINT_EVERY_TIMESTEPS = 15_000   # coarse enough that diagnostics
    CHECKPOINT_DIR = "runs/1m_100epochs_v2"
    N_DIAGNOSTIC_EPISODES = 20


# ---------------------------------------------------------------------------
# Target placement
# ---------------------------------------------------------------------------
def compute_target_pos(start_position, distance, angle_deg, y_offset):
    theta = np.radians(angle_deg)
    dx = distance * np.cos(theta)
    dz = distance * np.sin(theta)
    return np.array([
        start_position[0] + dx,
        start_position[1] + y_offset,
        start_position[2] + dz,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Outcome-distribution diagnostic (trained-model version)
# ---------------------------------------------------------------------------
def diagnose_with_model(model, env, n_episodes,):
    outcomes = {"crash_or_oob": 0, "tilt": 0, "hit": 0, "timeout": 0}
    steps_survived = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        final_reward = 0.0

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
            action = mean.squeeze(0).numpy()
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, _ = env.step(action)
            step_count += 1
            done = terminated or truncated
            final_reward = reward

        steps_survived.append(step_count)

        # classify by the terminal reward value -- matches _compute_reward's
        # known terminal values (-10 crash/oob, -5 tilt, +15 hit)
        if truncated and not terminated:
            outcomes["timeout"] += 1
        elif final_reward >= 15.0:
            outcomes["hit"] += 1
        elif final_reward == -5.0:
            outcomes["tilt"] += 1
        elif final_reward == -10.0:
            outcomes["crash_or_oob"] += 1

    print(f"[diagnostic] outcomes over {n_episodes} eps:", outcomes)
    print(f"[diagnostic] avg steps survived: {np.mean(steps_survived):.1f}")
    return outcomes


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(model, timesteps_done, cfg: TrainConfig):
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    path = os.path.join(cfg.CHECKPOINT_DIR, f"ppo_stage1_{timesteps_done}.pt")
    torch.save(model.state_dict(), path)
    print(f"[diag] actor_log_std = {model.actor_log_std.detach().numpy()}")
    total_norm = sum(p.data.norm().item() for p in model.parameters())
    print(f"[diag] total param norm = {total_norm:.4f}")
    print(f"[checkpoint] saved {path}")



# ---------------------------------------------------------------------------
# Helper method helping to display the drone state in the last 10 eps
# ---------------------------------------------------------------------------
def calc_drone_state(drone_state_arr, n=10):
    if len(drone_state_arr) < n:
        return

    recent = drone_state_arr[-n:]
    count = len(recent)

    avg_pos = [
        sum(s.position.x for s in recent) / count,
        sum(s.position.y for s in recent) / count,
        sum(s.position.z for s in recent) / count,
    ]
    avg_vel = [
        sum(s.velocity.x for s in recent) / count,
        sum(s.velocity.y for s in recent) / count,
        sum(s.velocity.z for s in recent) / count,
    ]
    avg_orient = [
        sum(s.orientation.x for s in recent) / count,
        sum(s.orientation.y for s in recent) / count,
        sum(s.orientation.z for s in recent) / count,
        sum(s.orientation.w for s in recent) / count,
    ]
    avg_rotor_rpm = [
        sum(s.rotor_rpm[i] for s in recent) / count
        for i in range(len(recent[0].rotor_rpm))
    ]

    return {
        "position": avg_pos,
        "velocity": avg_vel,
        "orientation": avg_orient,
        "rotor_rpm": avg_rotor_rpm,
        "n_averaged": count,
    }


# ---------------------------------------------------------------------------
# Main training loop  chunked, resuming the SAME model each chunk
# ---------------------------------------------------------------------------
def train(cfg: TrainConfig):
    env = InterceptorDroneEnv()
    # env.target_pos = compute_target_pos(
    #     (0, 0,0), cfg.TARGET_DISTANCE, cfg.TARGET_ANGLE_DEG, cfg.TARGET_Y_OFFSET
    # )

    #to atleast hit something maybe
    env.target_pos=np.array([1,1,1],dtype=np.float32)
    print("target placed at:", env.target_pos)
    print("dorne placet at:", env.drone_state.position)

    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0])
    # optimizer =torch.optim.SGD(
    #     params=model.parameters(),
    #     lr=cfg.LR,
        
    # )
    timesteps_done = 0
    all_episode_rewards = []
    drone_state_arr = []

    while timesteps_done < cfg.TOTAL_TIMESTEPS:


        model, optimizer, episode_rewards = ppo_train(
            env,
            total_timesteps=cfg.CHECKPOINT_EVERY_TIMESTEPS,   # one chunk per call
            num_steps=cfg.NUM_STEPS,
            gamma=cfg.GAMMA, lam=cfg.LAM, lr=cfg.LR,
            model=model,                  #  resumes, not restarts
        )
        all_episode_rewards.extend(episode_rewards)
        drone_state_arr.append(env.drone_state)
        timesteps_done += cfg.CHECKPOINT_EVERY_TIMESTEPS

        recent = all_episode_rewards[-10:] if all_episode_rewards else [0]




        log.info(f"timesteps={timesteps_done}  avg_reward(last 10 eps)={sum(recent)/len(recent):.5f}")
        log.info(f"avg state of the drone in last 10 eps={calc_drone_state(drone_state_arr,10)}")
        log.info(f"episode_rewards={recent}")
        log.info(f"avg_std_log={model.actor_log_std.detach().numpy()}")


        save_checkpoint(model, timesteps_done, cfg)
        diagnose_with_model(model, env, cfg.N_DIAGNOSTIC_EPISODES)

    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    model = train(cfg)
    evaluate(model, InterceptorDroneEnv(), n_episodes=100)