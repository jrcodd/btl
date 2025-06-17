from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import (
    RenderMode, ObservationType, TissueDissectionEnv, ActionType
)

def make_env():
    """Create a single environment instance"""
    return TissueDissectionEnv(**env_kwargs)

# Parameters matching your training setup
parameters = ["RGBD", "2", "False", "False"]
observation_type = ObservationType[parameters[0]]
render_mode = RenderMode.HUMAN  # For visualization during replay

# Environment configuration - match your training setup exactly
env_kwargs = {
    "image_shape": (64, 64),
    "observation_type": observation_type,
    "action_type": ActionType.CONTINUOUS,
    "time_step": 0.01,
    "frame_skip": 10,
    "settle_steps": 10,
    "render_mode": render_mode,
    "reward_amount_dict": {
        "unstable_deformation": -2.0,
        "distance_cauter_border_point": -10.0,
        "delta_distance_cauter_border_point": -10.0,
        "cut_connective_tissue": 2.0,
        "cut_tissue": -1.0,
        "worspace_violation": -0.0,
        "state_limits_violation": -0.0,
        "rcm_violation_xyz": -0.1,
        "rcm_violation_rpy": -0.1,
        "collision_with_board": -0.1,
        "successful_task": 50.0,
    },
    "camera_reset_noise": None,
    "with_board_collision": True,
    "rows_to_cut": int(parameters[1]),
    "control_retraction_force": eval(parameters[3]),
    "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
}

# Load the trained model
model_path = "/hpc/home/klc130/work/setup/btl/sofa_zoo/sofa_zoo/envs/tissue_dissection/runs/image_based1M.pth"
model = PPO.load(model_path)
print(f"Model observation space: {model.observation_space.shape}")

# Create environment with the same preprocessing as training
# Single environment for replay (no need for multiple processes)
env = DummyVecEnv([make_env])

# Apply the same preprocessing as training
# For image observations, transpose from (H, W, C) to (C, H, W)
env = VecTransposeImage(env)

# Apply frame stacking to match model expectation
# Model expects 16 channels = 4 frames × 4 RGBD channels
frame_stack = 4
env = VecFrameStack(env, n_stack=frame_stack)

print(f"Final env observation space: {env.observation_space.shape}")

# Verify shapes match
if model.observation_space.shape != env.observation_space.shape:
    raise ValueError(f"Model expects {model.observation_space.shape} but env has {env.observation_space.shape}")

print("Starting replay...")

# Run the replay
obs = env.reset()
total_reward = 0
step_count = 0

while True:
    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, done, info = env.step(action)
    total_reward += reward[0]  # reward is array with one element
    step_count += 1
    
    # Render
    env.render()
    
    # Check if episode is done
    if done[0]:
        print(f"Episode finished! Steps: {step_count}, Total reward: {total_reward:.2f}")
        break
    
    # Safety break for very long episodes
    if step_count > 2000:
        print(f"Episode too long, stopping at step {step_count}")
        break

env.close()
print("Replay completed!")
