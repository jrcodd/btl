from stable_baselines3 import PPO
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import (
    RenderMode, ObservationType, TissueDissectionEnv, ActionType
)

parameters = ["RGBD", "2", "False", "False"]
observation_type = ObservationType[parameters[0]]
render_mode = RenderMode.HUMAN	

env_kwargs = {
    "image_shape": (64, 64),
    "observation_type": observation_type,
    "action_type": ActionType.CONTINUOUS,
    "time_step": 0.01,
    "frame_skip": 10,
    "settle_steps": 10,
    "render_mode": render_mode,
    "reward_amount_dict": {
        "unstable_deformation": -1.0,
        "distance_cauter_border_point": -10.0,
        "delta_distance_cauter_border_point": -10.0,
        "cut_connective_tissue": 0.5,
        "cut_tissue": -0.1,
        "worspace_violation": -0.0,
        "state_limits_violation": -0.0,
        "rcm_violation_xyz": -0.0,
        "rcm_violation_rpy": -0.0,
        "collision_with_board": -0.1,
        "successful_task": 50.0,
    },
    "camera_reset_noise": None,
    "with_board_collision": True,
    "rows_to_cut": int(parameters[1]),
    "control_retraction_force": eval(parameters[3]),
    "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
}

# Construct a temporary dummy env to inspect observation space
dummy_env = TissueDissectionEnv(**env_kwargs)
print("Observation space shape:", dummy_env.observation_space.shape)

model_path = "/hpc/home/klc130/work/setup/btl/sofa_zoo/sofa_zoo/envs/tissue_dissection/runs/image_based1M.pth"
model = PPO.load(model_path)

# Check if loaded model's obs space matches the dummy
if model.observation_space.shape != dummy_env.observation_space.shape:
    raise ValueError(f"Model expects {model.observation_space.shape} but env has {dummy_env.observation_space.shape}")

# Wrap env to match training conditions

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

# Reload model with the correct env wrapper

obs = vec_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
