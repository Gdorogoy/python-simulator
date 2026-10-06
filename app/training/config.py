"""Shared PPO / imitation hyperparameters used by the Isaac trainers. See app/training/docs.md#config."""
from app.reward_functions.rewards import GAMMA as REWARD_GAMMA

DEFAULT_DISTANCE_LOW = 3.0
DEFAULT_DISTANCE_HIGH = 10.0
DEFAULT_BC_CHECKPOINT_PATH = "app/control/pretrained_bc_dagger.pt"  # must match PARAMS hidden/num_hidden_layers

PARAMS = {
    'hidden': 64, 'num_hidden_layers': 4, 'dropout': 0.0,  # dropout must stay 0, see docs.md#config
    'lr': 5e-5, 'gamma': REWARD_GAMMA, 'lam': 0.97,
    'clip_eps': 0.285, 'vf_coef': 0.525, 'target_kl': 0.0245,
    'num_epochs': 10, 'num_minibatches': 64, 'max_grad_norm': 0.25,
    'log_std_max': -0.9,
}

IMITATION_RETRAIN_EPOCHS = 5
IMITATION_BC_LR = 3e-4
IMITATION_BC_BATCH_SIZE = 4096
IMITATION_BUFFER_CAP_PAIRS = 1_000_000  # aggregate (obs, pid_action) pairs kept across imitation rounds
RECENCY_DECAY = 0.85  # a pair from k blocks ago is kept/weighted with RECENCY_DECAY**k
N_DIAGNOSTIC_EPISODES = 10
WEIGHT_DECAY = 1e-4
LR_MIN_RATIO = 0.01  # cosine lr floor as a fraction of PARAMS['lr']
ENT_COEF_START = 0.01
ENT_COEF_END = 0.001
