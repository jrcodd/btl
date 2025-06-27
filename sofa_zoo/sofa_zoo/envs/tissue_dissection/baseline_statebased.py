import argparse
import os
from distutils.util import strtobool
import time

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from stable_baselines3 import PPO
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import RenderMode, ObservationType, TissueDissectionEnv, ActionType
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

def setup_environment(use_wandb: bool = False):
    add_render_callback = False
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf
    parameters = ["STATE", "2", "False", "False"] 
    observation_type = ObservationType[parameters[0]]
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]
    render_mode = RenderMode.NONE
    env_kwargs = {
        "image_shape": (64, 64),
        "window_size": (600, 600),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 10,
        "render_mode": render_mode, 
        "reward_amount_dict": {
            "unstable_deformation": -1.0, #original -1.0
            "distance_cauter_border_point": -10.0, #original -10.0
            "delta_distance_cauter_border_point": -10.0, #original -10.0
            "cut_connective_tissue": 0.5, #original 0.5
            "cut_tissue": -0.1, #original -0.1
            "worspace_violation": -0.0, #original -0.0
            "state_limits_violation": -0.0, #original -0.0
            "rcm_violation_xyz": -0.0, #original -0.0
            "rcm_violation_rpy": -0.0, #original -0.0
            "collision_with_board": -0.1, #original -0.1
            "successful_task": 50.0, #original 50.0
        },
        "camera_reset_noise": None,
        "with_board_collision": True,
        "rows_to_cut": int(parameters[1]),
        "control_retraction_force": eval(parameters[3]),
        "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
    }

    config = {"max_episode_steps": max(600, env_kwargs["rows_to_cut"] * 200), **CONFIG}

    if image_based:
        ppo_kwargs = PPO_KWARGS["image_based"]
    else:
        ppo_kwargs = PPO_KWARGS["state_based"]
    info_keywords = [
        "ret", "ret_col_wit_boa", "ret_cut_con_tis", "ret_cut_tis",
        "ret_del_dis_cau_bor_poi", "ret_dis_cau_bor_poi", "ret_num_tet_in_con_tis",
        "ret_num_tet_in_tis", "ret_rcm_vio_rot", "ret_rcm_vio_xyz", "ret_sta_lim_vio",
        "ret_suc_tas", "ret_uns_def", "ret_wor_vio", "unstable_deformation",
        "successful_task", "total_cut_connective_tissue", "total_cut_tissue",
        "total_unstable_deformation", "ratio_cut_connective_tissue", "ratio_cut_tissue",
    ]

    config["ppo_config"] = ppo_kwargs
    config["env_kwargs"] = env_kwargs
    config["info_keywords"] = info_keywords

    model, callback = configure_learning_pipeline(
        env_class=TissueDissectionEnv,
        env_kwargs=env_kwargs,
        pipeline_config=config,
        monitoring_keywords=info_keywords,
        normalize_observations=False if image_based else True,
        algo_class=PPO,
        algo_kwargs=ppo_kwargs,
        render=add_render_callback,
        use_wandb=use_wandb,
        reward_clip=reward_clip,
        normalize_reward=normalize_reward,
        use_watchdog_vec_env=True,
        watchdog_vec_env_timeout=30.0,
        reset_process_on_env_reset=True,
    )
    return model, callback
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='dissect_experiment', help='Name of the experiment for logging')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--total-timesteps', type=int, default=1000000, help='Total number of timesteps for training')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, torch.backends.cudnn.deterministic=False')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True, help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--track', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=True, help='if toggled, this experiment will be tracked with weights and biases')
    parser.add_argument('--wandb-project-name', type=str, default='dissect_experiment', help='Weights and Biases project name')
    parser.add_argument('--wandb-entity', type=str, default=None, help='the entity (team) of wandb\'s project')

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"dissect_{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            name=run_name,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
    )
    model, callback = setup_environment(args.track)
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name=run_name,
    )
    model.save(f"models/{run_name}")
    print(f"Experiment logged to runs/{run_name}")
