import pkg_resources
from subprocess import callfor dist in pkg_resources.working_set:
    call("python -m pip install --upgrade " + dist.<projectname>, shell=True)