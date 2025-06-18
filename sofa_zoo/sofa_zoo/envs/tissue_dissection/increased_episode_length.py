import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import RenderMode, ObservationType, TissueDissectionEnv, ActionType
from sofa_zoo.common.sb3_setup import configure_learning_pipeline
from sofa_zoo.common.lapgym_experiment_parameters import CONFIG, PPO_KWARGS

class PeriodicSaveCallback(BaseCallback):
    """
    Callback for saving the model periodically during training
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "model", verbose: int = 0):
        super(PeriodicSaveCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        
        # Create save directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.n_calls}_steps")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Saved model at {self.n_calls} steps to {model_path}")
        return True

def create_env_with_wrappers(env_kwargs, image_based=True, frame_stack=4):
    """Create environment with proper wrappers for image-based training"""
    def make_env():
        # Filter out window_size if it exists and TissueDissectionEnv doesn't support it
        filtered_kwargs = {k: v for k, v in env_kwargs.items() if k != 'window_size'}
        return TissueDissectionEnv(**filtered_kwargs)
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    if image_based:
        # Apply image preprocessing
        env = VecTransposeImage(env)
        env = VecFrameStack(env, n_stack=frame_stack)
    
    return env

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
    render_mode = RenderMode.HUMAN
    
    # Environment configuration - avoid window_size as it's not supported by TissueDissectionEnv
    env_kwargs = {
        # 3. Define the image resolution. (64, 64) is a common size for faster training.
        "image_shape": (64, 64),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 10,
        "render_mode": render_mode, 
        "reward_amount_dict": {
            "unstable_deformation": -10.0, #original 1.0
            "distance_cauter_border_point": -6.0, #original -10.0
            "delta_distance_cauter_border_point": -6.0, #original -10.0
            "cut_connective_tissue": 2.5, #original 0.5
            "cut_tissue": -0.8, #original -0.1
            "worspace_violation": -0.0, #original -0.0
            "state_limits_violation": -0.0, #original -0.0
            "rcm_violation_xyz": -0.1, #original -0.0
            "rcm_violation_rpy": -0.1, #original -0.0
            "collision_with_board": -0.1, #original -0.1
            "successful_task": 50.0, #original 50.0
        },
        "camera_reset_noise": None,
        "with_board_collision": True,
        "rows_to_cut": int(parameters[1]),
        "control_retraction_force": eval(parameters[3]),
        "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
    }

    config = {"max_episode_steps": 1024, **CONFIG} #1024 timesteps to try and get model to not stall after 10 seconds.

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

    # Setup paths and model loading
    log_path = os.path.expanduser("~/work/dissect_models/")
    os.makedirs(log_path, exist_ok=True)
    
    starting_model_path = "/hpc/home/klc130/work/dissect_models/PPO_RGBD_2rows_400000_steps.zip"
    
    # Check if we should load a pre-trained model or create from scratch
    if os.path.exists(starting_model_path):
        print("Loading pre-trained model from:", starting_model_path)
        
        try:
            # Method 1: Load model and create environment manually
            # This avoids the window_size issue in configure_learning_pipeline
            model = PPO.load(starting_model_path)
            print("Successfully loaded 3M starting model")
            
            # Create environment with proper wrappers
            env = create_env_with_wrappers(env_kwargs, image_based=image_based)
            model.set_env(env)
            
            # Create a simple callback
            callback = CallbackList([])
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to creating new model")
            
            # Method 2: Use configure_learning_pipeline (but add window_size for compatibility)
            config["ppo_config"] = ppo_kwargs
            config["env_kwargs"] = {**env_kwargs, "window_size": (64, 64)}  # Add for sb3_setup compatibility
            config["info_keywords"] = info_keywords
            
            model, callback = configure_learning_pipeline(
                env_class=TissueDissectionEnv,
                env_kwargs=config["env_kwargs"],
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
    else:
        print("Starting model not found, creating new model and training from scratch")
        
        # Use configure_learning_pipeline for new model creation
        config["ppo_config"] = ppo_kwargs
        config["env_kwargs"] = {**env_kwargs, "window_size": (64, 64)}  # Add for sb3_setup compatibility
        config["info_keywords"] = info_keywords
        
        model, callback = configure_learning_pipeline(
            env_class=TissueDissectionEnv,
            env_kwargs=config["env_kwargs"],
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
    
    # Configuration for periodic saving
    save_frequency = 50000  # Save every 50k steps
    total_training_iterations = 10
    timesteps_per_iteration = config.get("total_timesteps", 100000)
    
    # Create periodic save callback
    save_callback = PeriodicSaveCallback(
        save_freq=save_frequency,
        save_path=log_path,
        name_prefix=f"PPO_{observation_type.name}_{parameters[1]}rows",
        verbose=1
    )
    
    # Combine callbacks
    combined_callback = CallbackList([callback, save_callback])
    
    print(f"Starting training for {total_training_iterations} iterations")
    print(f"Each iteration: {timesteps_per_iteration} timesteps")
    print(f"Models will be saved every {save_frequency} steps to: {log_path}")
    
    # Training loop with periodic major saves
    for i in range(1, total_training_iterations + 1):
        print(f"\n=== Starting Training Iteration {i}/{total_training_iterations} ===")
        
        model.learn(
            total_timesteps=timesteps_per_iteration,
            callback=combined_callback,
            tb_log_name=f"PPO_{observation_type.name}_{parameters[1]}rows_{parameters[2]}vis_iter{i}",
            reset_num_timesteps=False  # Continue counting timesteps across iterations
        )
        
        # Save major checkpoint after each iteration
        iteration_save_path = os.path.join(log_path, f"model_iteration_{i}.zip")
        model.save(iteration_save_path)
        print(f"Saved iteration {i} model to: {iteration_save_path}")
    
    # Final save
    final_save_path = os.path.join(log_path, "final_model.zip")
    model.save(final_save_path)
    print(f"Training complete! Final model saved to: {final_save_path}")
