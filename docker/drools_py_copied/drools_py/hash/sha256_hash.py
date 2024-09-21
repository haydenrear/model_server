import hashlib
import os
import subprocess


def compute_sha256_hash(file_path: str):
    hasher = hashlib.sha256()
    if not os.path.exists(file_path):
        return ''
    with open(file_path, 'rb') as f:
        if len(f.readlines()) != 0:
            for chunk in f.read():
                hasher.update(chunk)

    return hasher.hexdigest()


def get_git_hash(file_name: str):
    if not os.path.exists(file_name):
        return ''
    with open(file_name, 'r') as open_file:
        for line in open_file.readlines():
            if "sha256" in line:
                return line.split("sha256:")[1]


def download_git_hashes(git_repo: str, git_hash_dir: str):
    if not os.path.exists(git_hash_dir):
        subprocess.run(["git", "clone", git_repo, git_hash_dir], env=dict({"GIT_LFS_SKIP_SMUDGE": "1"}))
    # TODO: copy to temp to make sure the same
