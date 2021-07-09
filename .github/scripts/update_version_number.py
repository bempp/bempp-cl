import json
import sys
from datetime import datetime
from github import Github

access_key = sys.argv[-2]
branch_name = sys.argv[-1]

if not branch_name.startswith("refs/tags"):
    git = Github(access_key)

    bempp = git.get_repo("bempp/bempp-cl")
    branch = bempp.get_branch(branch_name)

    version = (0,)

    vfile1 = bempp.get_contents("VERSION", branch.commit.sha)
    v1 = tuple(int(i) for i in vfile1.decoded_content.split(b"."))

    vfile2 = bempp.get_contents("codemeta.json", branch.commit.sha)
    data = json.loads(vfile2.decoded_content)
    v2 = tuple(int(i) for i in data["version"][1:].split("."))
    assert v2 == v1

    vfile3 = bempp.get_contents("bempp/version.py", branch.commit.sha)
    v3 = tuple(
        int(i)
        for i in vfile3.decoded_content.split(b"\n", 1)[1].split(b'"')[1].split(b".")
    )
    assert v3 == v1
