from torch import nn
from sofa_zoo.models.cnn import ActorCriticDoubleNatureCnnPolicy
from sofa_zoo.common.schedules import linear_schedule

CONFIG = {
    "total_timesteps": int(1e7), # 10 million timesteps is default going to 1 million for testing purposes
    "number_of_envs": 10, #going back to 8 since pc froze at 14
    "checkpoint_distance": int(5e5),
    "frame_stack": 4, #4 is default will try to stack more frames to see if it works better in the future
    "videos_per_run": 0,
    "video_length": 300,
}

PPO_KWARGS = {
    "image_based": {
        "policy": ActorCriticDoubleNatureCnnPolicy,
        "n_steps": 128,  # SB3 nature default: 128, SB3 default: 2048
        "batch_size": 256,  # SB3 nature default: 256, SB3 default: 64
        "learning_rate": linear_schedule(2.5e-4),  # SB3 nature default: linear_schedule(2.5e-4)
        "n_epochs": 4,  # SB3 nature default: 4, SB3 default: 10
        "gamma": 0.995,  # SB3 nature default: 0.99
        "gae_lambda": 0.95, 
        "clip_range": linear_schedule(0.1),  # SB3 nature default: linear_schedule(0.1)
        "clip_range_vf": 0.2,  # SB3 nature default: 0.2
        "ent_coef": 0.0,  # SB3 nature default: 0.01
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
    },
    "state_based": {
        "policy": "MlpPolicy",
        "n_steps": 128,  # SB3 nature default: 128, SB3 default: 2048
        "batch_size": 256,  # SB3 nature default: 256, SB3 default: 64
        "learning_rate": linear_schedule(2.5e-4),  # SB3 nature default: linear_schedule(2.5e-4)
        "n_epochs": 4,  # SB3 nature default: 4, SB3 default: 10
        "gamma": 0.995,  # SB3 nature default: 0.99
        "gae_lambda": 0.95,
        "clip_range": linear_schedule(0.1),  # SB3 nature default: linear_schedule(0.1)
        "clip_range_vf": 0.2,  # SB3 nature default: 0.2
        "ent_coef": 0.0,  # SB3 nature default: 0.01
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "policy_kwargs": dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
        ),
    },
}
