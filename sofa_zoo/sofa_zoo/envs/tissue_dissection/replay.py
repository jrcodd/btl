import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import RenderMode, ObservationType, TissueDissectionEnv, ActionType

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

def evaluate_model(model_path, env_kwargs, n_episodes=5, render=True, deterministic=True):
    """
    Evaluate a trained model by running episodes
    
    Args:
        model_path: Path to the saved model (.zip file)
        env_kwargs: Environment configuration dictionary
        n_episodes: Number of episodes to run
        render: Whether to show visual rendering
        deterministic: Whether to use deterministic policy (no exploration)
    """
    
    # Load the trained model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    
    # Determine if this is image-based
    observation_type = env_kwargs.get("observation_type", ObservationType.STATE)
    image_based = observation_type in [ObservationType.RGB, ObservationType.RGBD]
    
    # Create environment with same configuration as training
    env = create_env_with_wrappers(env_kwargs, image_based=image_based)
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        print(f"\n=== Episode {episode + 1}/{n_episodes} ===")
        
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=deterministic)
            
            # Take action in environment
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]  # reward is a list with one element
            episode_length += 1
            
            # Check if task was successful (if this info is available)
            if len(info) > 0 and 'successful_task' in info[0]:
                if info[0]['successful_task']:
                    success_count += 1
                    print("Task completed successfully!")
            
            # Optional: print step info
            if episode_length % 50 == 0:
                print(f"Step {episode_length}, Reward: {reward[0]:.3f}, Total: {episode_reward:.3f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1} finished:")
        print(f"  Total Reward: {episode_reward:.3f}")
        print(f"  Episode Length: {episode_length}")
        
        # Print additional info if available
        if len(info) > 0:
            info_dict = info[0]
            relevant_keys = ['successful_task', 'total_cut_connective_tissue', 'total_cut_tissue', 
                           'ratio_cut_connective_tissue', 'ratio_cut_tissue']
            for key in relevant_keys:
                if key in info_dict:
                    print(f"  {key}: {info_dict[key]}")
    
    # Print summary statistics
    print(f"\n=== Evaluation Summary ===")
    print(f"Episodes run: {n_episodes}")
    print(f"Average reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Reward range: {np.min(episode_rewards):.3f} to {np.max(episode_rewards):.3f}")
    
    env.close()
    return episode_rewards, episode_lengths, success_count

if __name__ == "__main__":
    # Configuration - should match your training configuration
    parameters = ["RGBD", "2", "False", "False"] 
    observation_type = ObservationType[parameters[0]]
    
    # Set render_mode to HUMAN to see the visualization
    render_mode = RenderMode.HUMAN  # Change to HEADLESS if you don't want visualization
    
    env_kwargs = {
        "image_shape": (64, 64),
        "observation_type": observation_type,
        "action_type": ActionType.CONTINUOUS,
        "time_step": 0.01,
        "frame_skip": 10,
        "settle_steps": 10,
        "render_mode": render_mode, 
        "reward_amount_dict": {
            "unstable_deformation": -10.0,
            "distance_cauter_border_point": -1.0,
            "delta_distance_cauter_border_point": -1.0,
            "cut_connective_tissue": 5.0,
            "cut_tissue": -3.5,
            "worspace_violation": -0.0,
            "state_limits_violation": -0.0,
            "rcm_violation_xyz": -0.3,
            "rcm_violation_rpy": -0.3,
            "collision_with_board": -0.1,
            "successful_task": 75.0,
        },
        "camera_reset_noise": None,
        "with_board_collision": True,
        "rows_to_cut": int(parameters[1]),
        "control_retraction_force": eval(parameters[3]),
        "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
    }
    
    # Path to your saved model - UPDATE THIS PATH
    model_path = os.path.expanduser("~/work/dissect_models/PPO_RGBD_2rows_100000_steps.zip")
    
    # Alternative paths you might want to try:
    # model_path = os.path.expanduser("~/work/dissect_models/model_iteration_1.zip")
    # model_path = os.path.expanduser("~/work/dissect_models/PPO_RGBD_2rows_450000_steps.zip")
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print("Available models in directory:")
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.zip'):
                    print(f"  {os.path.join(model_dir, file)}")
    else:
        # Run evaluation
        evaluate_model(
            model_path=model_path,
            env_kwargs=env_kwargs,
            n_episodes=3,  # Number of episodes to run
            render=True,   # Set to False if you don't want visualization
            deterministic=True  # Set to False if you want stochastic policy
        )
