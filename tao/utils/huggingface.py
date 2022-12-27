from pathlib import Path
import os

from huggingface_hub import HfApi, HfFolder, Repository
from huggingface_hub.repocard import metadata_eval_result, metadata_save

import torch
import datetime
import json
import shutil
import numpy as np

def dump_params(model, path):
    o_params = model.__dict__
    n_params = dict()
    for (key, value) in o_params.items():
        if key[0] != "_" and key !="envs":
            n_params[key] = value
            
    with open(Path(path)/"hyperparameters.json", "w") as outfile:
        json.dump(n_params, outfile)
        
def push_to_hub(model, repo_id, local_repo_path, episodic_returns):
    _, repo_name = repo_id.split("/")

    api = HfApi()
    env_name = model.env_id

    repo_url = api.create_repo(
        repo_id=repo_id,
        token=None,
        private=False,
        exist_ok=True,)

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path, clone_from=repo_url, use_auth_token=True)
    repo.git_pull()

    repo.lfs_track(["*.mp4"])

    # Step 1: Save the model
    torch.save(model.state_dict(), os.path.join(repo_local_path,"model.pt"))

    # Step 2: Save the hyperparameters to JSON
    dump_params(model, repo_local_path)

    mean_reward, std_reward = np.average(episodic_returns), np.std(episodic_returns)

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "env_id": env_name, 
        "mean_reward": float(mean_reward),
        "n_evaluation_episodes": len(episodic_returns),
        "eval_datetime": eval_form_datetime,
    }
    # Write a JSON file
    with open(Path(repo_local_path) / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    metadata = {}
    metadata["tags"] = [
        env_name,
        "reinforce",
        "reinforcement-learning",
        "custom-implementation",
        "deep-rl-class"
    ]

    # Add metrics
    eval = metadata_eval_result(
    model_pretty_name=repo_name,
    task_pretty_name="reinforcement-learning",
    task_id="reinforcement-learning",
    metrics_pretty_name="mean_reward",
    metrics_id="mean_reward",
    metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
    dataset_pretty_name=env_name,
    dataset_id=env_name,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
    # Agent playing **{env_name}**
    """

    readme_path = repo_local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 4: Record a video
    video_folder_path = Path("./videos") / model.run_name
    video_files = list(video_folder_path.glob("*.mp4"))
    latest_file = max(video_files, key=lambda file: int("".join(filter(str.isdigit, file.stem))))
    shutil.copyfile(latest_file, Path(repo_local_path) / "replay.mp4")

    # Push everything to hub
    print(f"Pushing repo {repo_name} to the Hugging Face Hub")
    repo.push_to_hub(commit_message="run trail")