#!/usr/bin/env python3
#
# Install this script's dependencies with pip3:
# pip3 install huggingface-hub


from huggingface_hub import snapshot_download
import os


repos = [
    "BAAI/bge-reranker-v2-m3",
]

def download_model(repo_id):
    local_dir = os.path.abspath(os.path.join("huggingface.co", repo_id))
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)


if __name__ == "__main__":
    for repo_id in repos:
        print(f"Downloading huggingface repo {repo_id}...")
        download_model(repo_id)