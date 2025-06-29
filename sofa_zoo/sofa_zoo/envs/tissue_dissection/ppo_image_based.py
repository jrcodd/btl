import numpy as np
from stable_baselines3 import PPO
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import RenderMode, ObservationType, TissueDissectionEnv, ActionType
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

if __name__ == "__main__":
    add_render_callback = False
    continuous_actions = True
    normalize_reward = True
    reward_clip = np.inf
    
    # 1. Change the observation type from "STATE" to "RGBD" for image based
    parameters = ["RGBD", "2", "False", "False"] 
    observation_type = ObservationType[parameters[0]]
    
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]
    
    # 2. Set the render_mode to HEADLESS (will not show a gui but the system will still get image data)
    # Image observations cannot be generated with RenderMode.NONE.
    render_mode = RenderMode.HEADLESS
    
    env_kwargs = {
        # 3. Define the image resolution. (64, 64) is a common size for faster training.
        "image_shape": (64, 64),
        "window_size": (400, 400),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS if continuous_actions else ActionType.DISCRETE,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 10,
        "render_mode": render_mode, 
        "reward_amount_dict": {
            "unstable_deformation": -10.0, #original 1.0
            "distance_cauter_border_point": -1.0, #original -10.0
            "delta_distance_cauter_border_point": -1.0, #original -10.0
            "cut_connective_tissue": 5.0, #original 0.5
            "cut_tissue": -3.5, #original -0.1
            "worspace_violation": -0.0, #original -0.0
            "state_limits_violation": -0.0, #original -0.0
            "rcm_violation_xyz": -0.1, #original -0.0
            "rcm_violation_rpy": -0.1, #original -0.0
            "collision_with_board": -0.1, #original -0.1
            "successful_task": 75.0, #original 50.0
        },
        "camera_reset_noise": None,
        "with_board_collision": True,
        "rows_to_cut": int(parameters[1]),
        "control_retraction_force": eval(parameters[3]),
        "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
    }

    config = {"max_episode_steps": max(400, env_kwargs["rows_to_cut"] * 200), **CONFIG}

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
        reward_clip=reward_clip,
        normalize_reward=normalize_reward,
        use_watchdog_vec_env=True,
        watchdog_vec_env_timeout=30.0,
        reset_process_on_env_reset=True,
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback,
        tb_log_name= f"PPO_{observation_type.name}_{parameters[1]}rows_{parameters[2]}vis",
    )
    
    
    model.save(log_path + "saved_model.pth")
