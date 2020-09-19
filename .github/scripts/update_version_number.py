import json
import sys
from datetime import datetime
from github import Github

access_key = sys.argv[-2]
branch_name = sys.argv[-1]

git = Github(access_key)

bempp = git.get_repo("bempp/bempp-cl")
branch = bempp.get_branch(branch_name)

version = (0, )

vfile1 = bempp.get_contents("VERSION", branch.commit.sha)
v1 = tuple(int(i) for i in vfile1.decoded_content.split(b"."))
if v1 > version:
    version = v1

vfile2 = bempp.get_contents("codemeta.json", branch.commit.sha)
data = json.loads(vfile2.decoded_content)
v2 = tuple(int(i) for i in data["version"][1:].split("."))
if v2 > version:
    version = v2

vfile3 = bempp.get_contents("bempp/version.py", branch.commit.sha)
v3 = tuple(int(i) for i in vfile3.decoded_content.split(b'"')[1].split(b"."))
if v3 > version:
    version = v3

v_str = ".".join(str(i) for i in version)
if v1 != version:
    bempp.update_file("VERSION", "Update version number", v_str + "\n", sha=vfile1.sha, branch=branch_name)
if v2 != version:
    data["version"] = "v" + v_str
    data["dateModified"] = datetime.now().strftime("%Y-%m-%d")
    bempp.update_file("codemeta.json", "Update version number", json.dumps(data), sha=vfile2.sha, branch=branch_name)
if v3 != version:
    bempp.update_file("bempp/version.py", "Update version number", "__version__ = \"" + v_str + "\"\n" , sha=vfile3.sha, branch=branch_name)
