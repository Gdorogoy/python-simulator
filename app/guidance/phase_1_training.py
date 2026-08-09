"""
###!!! 
Only teaching the drone how to hover so it wont fall 

"""
from datetime import datetime

import numpy as np
import torch
import os

import matplotlib as plt
from app.environmental.interceptor_drone import InterceptorDroneEnv
from app.guidance.train import ActorCritic, ppo_train, evaluate

import logging
plt.use("Agg") 

os.makedirs("prod_logs", exist_ok=True)
log_filename = f"prod_logs/phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
    NUM_STEPS = 7500                   # rollout length per PPO update
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
def diagnose_with_model(model, env, n_episodes):
    outcomes = {"oob": 0, "attitude": 0, "hit": 0, "moving_away_cap": 0, "timeout": 0}
    steps_survived = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        step_count = 0
        last_reason = None

        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                mean, std, _ = model.forward(obs_t)
            action = np.clip(mean.squeeze(0).numpy(), env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
            last_reason = info["reason"]

        steps_survived.append(step_count)

        if truncated and not terminated:
            outcomes["timeout"] += 1
        else:
            outcomes[last_reason] += 1

    print(f"[diagnostic] outcomes over {n_episodes} eps:", outcomes)
    print(f"[diagnostic] avg steps survived: {np.mean(steps_survived):.1f}")
    print(f"step {step_count}: raw_mean={mean.squeeze(0).numpy()}")
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
    if len(drone_state_arr) < 0:
        return


    elif len(drone_state_arr)< n: 

        
        recent = drone_state_arr[-len(drone_state_arr)-1:]
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

    else:


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



#----------------------------------------------------------------------------
# Graph for the distance
#----------------------------------------------------------------------------
def show_graph(dist_history):
    plt.figure(figsize=(8, 4))
    plt.plot(dist_history)
    plt.xlabel("Step")
    plt.ylabel("Distance to target (m)")
    plt.title("Drone distance from target over one episode")
    plt.axhline(y=0.3, color='r', linestyle='--', label='hit threshold (0.3m)')
    plt.legend()
    plt.savefig("distance_trace.png")
    plt.close()


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
    dist_history=[]

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

    show_graph(dist_history)

    return model


if __name__ == "__main__":
    cfg = TrainConfig()
    model = train(cfg)
    evaluate(model, InterceptorDroneEnv(), n_episodes=100)