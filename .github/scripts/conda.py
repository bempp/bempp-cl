import os
import sys
import github
import hashlib

access_key = sys.argv[-1]

git = github.Github(access_key)

symfem = git.get_repo("bempp/bempp-cl")
branch = symfem.get_branch("master")

version = (
    symfem.get_contents("VERSION", branch.commit.sha).decoded_content.decode().strip()
)


os.system(f"wget https://github.com/bempp/bempp-cl/archive/v{version}.tar.gz")

sha256_hash = hashlib.sha256()
with open(f"v{version}.tar.gz", "rb") as f:
    # Read and update hash string value in blocks of 4K
    for byte_block in iter(lambda: f.read(4096), b""):
        sha256_hash.update(byte_block)
    hash = sha256_hash.hexdigest()

upstream_feedstock = git.get_repo("conda-forge/bempp-cl-feedstock")
upstream_branch = upstream_feedstock.get_branch("master")

fork = git.get_user().create_fork(upstream_feedstock)

u = git.get_user()

for repo in u.get_repos():
    if repo.full_name.startswith("bemppbot/bempp-cl-feedstock"):
        repo.delete()

fork = git.get_user().create_fork(upstream_feedstock)
branch = fork.get_branch("master")

old_meta = fork.get_contents("recipe/meta.yaml", branch.commit.sha)

old_meta_lines = old_meta.decoded_content.decode().split("\n")
new_meta_lines = []
for line in old_meta_lines:
    if line.startswith("{% set version"):
        new_meta_lines.append(f'{{% set version = "{version}" %}}')
    elif "sha256" in line:
        newline = line.split("sha256")[0]
        newline += f"sha256: {hash}"
        new_meta_lines.append(newline)
    elif "number" in line:
        newline = line.split("number")[0]
        newline += "number: 0\n"
        new_meta_lines.append(newline)
    else:
        new_meta_lines.append(line)

fork.update_file(
    "recipe/meta.yaml", "Update version", "\n".join(new_meta_lines), sha=old_meta.sha
)

upstream_feedstock.create_pull(
    title=f"Update version to {version}",
    body="",
    base="master",
    head="bemppbot:master",
)
