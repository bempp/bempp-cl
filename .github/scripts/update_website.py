import sys
from github import Github

access_key = sys.argv[-1]


git = Github(access_key)

bempp = git.get_repo("bempp/bempp-cl")
tag = bempp.get_tags()[0].name
changelog = bempp.get_release(tag).body

repo = git.get_repo("bempp/bempp-website")

main = repo.get_branch("main")

new_config = []
config = repo.get_contents("_config.yml")
for line in config.decoded_content.decode("utf8").split("\n"):
    if line.startswith("bemppversion:"):
        new_config.append("bemppversion: Bempp-cl " + tag[1:])
    else:
        new_config.append(line)

repo.update_file(
    "_config.yml", "Update version number", "\n".join(new_config), sha=config.sha
)

old_changelog = repo.get_contents("changelog.md")

new_changelog = "---\ntitle: Changelog\n---\n"
new_changelog += "## [Bempp-cl " + tag[1:] + "]"
new_changelog += "(https://github.com/bempp/bempp-cl/releases/tag/" + tag + ")\n"
new_changelog += changelog.replace("\r\n", "\n")
new_changelog += old_changelog.decoded_content.decode("utf8").split("---", 3)[2]

repo.update_file(
    "changelog.md", "Add to changelog", new_changelog, sha=old_changelog.sha
)
