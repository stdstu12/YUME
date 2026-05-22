"""Download Yume model weights from HuggingFace."""

from huggingface_hub import snapshot_download

REPOS = [
    "stdstu123/Yume-5B-720P",
    "OpenGVLab/InternVL3-2B-Instruct",
]

for repo_id in REPOS:
    local_dir = f"./{repo_id.split('/')[-1]}"
    print(f"Downloading {repo_id} -> {local_dir}")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Done: {local_dir}")
