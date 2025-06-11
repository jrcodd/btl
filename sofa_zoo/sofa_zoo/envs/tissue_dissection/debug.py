import cv2
import numpy as np
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from sofa_env.scenes.tissue_dissection.tissue_dissection_env import (
    RenderMode, ObservationType, TissueDissectionEnv, ActionType
)

parameters = ["STATE", "2", "False", "False"]
observation_type = ObservationType.STATE

render_mode = RenderMode.HUMAN

env_kwargs = {
    "observation_type": observation_type,
    "action_type": ActionType.CONTINUOUS,
    "time_step": 0.01,
    "frame_skip": 10,
    "settle_steps": 10,
    "render_mode": render_mode,
    "image_shape": (800, 800),
    "reward_amount_dict": {
        "unstable_deformation": -1.0,
        "distance_cauter_border_point": -10.0,
        "delta_distance_cauter_border_point": -10.0,
        "cut_connective_tissue": 0.5,
        "cut_tissue": -0.1,
        "successful_task": 50.0,
    },
    "with_board_collision": True,
    "rows_to_cut": int(parameters[1]),
    "create_scene_kwargs": {"show_border_point": eval(parameters[2])},
}

env = make_vec_env(lambda: TissueDissectionEnv(**env_kwargs), n_envs=1)
env = VecFrameStack(env, n_stack=4)
model_path = "runs/PPO_STATE_2rows_Falsevis_1/PPO_STATE_2rows_Falsevis_1saved_model.pth"
model = PPO.load(model_path, env=env)

print("Running simulation and collecting frames...")
obs = env.reset()
done = False
frames = []
max_steps = 500  

for step in range(max_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    frame = env.render()
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frames.append(bgr_frame)

    if done.any():
        print("Episode finished.")
        break

env.close()

if frames:
    output_directory = Path("runs/videos")
    output_directory.mkdir(exist_ok=True)
    video_path = output_directory / "tissue_dissection_video.mp4"
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30 
    
    video_writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    print(f"Saving video to {video_path}...")
    for frame in frames:
        video_writer.write(frame)

    video_writer.release()
    print("Video saved successfully.")
else:
    print("No frames were collected. No video will be saved.")
