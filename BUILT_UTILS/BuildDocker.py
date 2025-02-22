import subprocess
import os

print(os.getcwd())

dockerhub_repo = 'repo_name'
dockerhub_tag = 'latest'
dockerhub_username = 'repo_uname'


build_command = f'docker build -t {dockerhub_repo}:{dockerhub_tag} .'
subprocess.run(build_command, shell=True, check=True)

tag_command = f'docker tag {dockerhub_repo}:{dockerhub_tag} {dockerhub_username}/{dockerhub_repo}:{dockerhub_tag}'
subprocess.run(tag_command, shell=True, check=True)

push_command = f'docker push {dockerhub_username}/{dockerhub_repo}:{dockerhub_tag}'
subprocess.run(push_command, shell=True, check=True)
