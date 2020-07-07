#!/usr/bin/env python3
import argparse
import os
import pkg_resources
import subprocess


SYNBOLS_INCLUDE_PATH = os.path.join(pkg_resources.require("synbols")[0].location, "synbols")
SYNBOLS_VERSION = pkg_resources.require("synbols")[0].version
DOCKER_IMAGE = "aldro61/synbols_dev"
DOCKER_TAG = "v%s" % SYNBOLS_VERSION  # XXX: the tag matches the package version


def is_docker_installed():
    """Check if Docker is installed

    Returns
    -------
    bool
        True if the docker is installed, False otherwise.
    """
    try:
        subprocess.Popen(["docker"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate()
        return True
    except FileNotFoundError:
        return False


def is_docker_image_available():
    """Check if the Synbols docker image is available on the local machine

    Returns
    -------
    bool
        True if the image is available, False otherwise.
    """
    stdout, stderr = subprocess.Popen(["docker", "images", "-q", "%s:%s" % (DOCKER_IMAGE, DOCKER_TAG)],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT).communicate()
    return stdout != ""


def run_in_docker(file, paths, args):
    """
    Run a Python script with the Synbols Docker image

    Parameters
    ----------
    file : str
        The Python script to run
    paths: list of str
        Paths to be made acessible at run time
    args : list
        A list of command line arguments to pass to the Python script

    """
    # Generate docker arguments to mount all expected directories
    paths = [] if paths is None else paths
    curdir = os.path.abspath(os.getcwd())
    path_mounts = " ".join([f"-v {p}:{p}" for p in paths + [curdir]])

    # Merge all command line arguments
    args = " ".join(args)

    # Generate and run the docker command
    cmd = f"docker run --rm --user {os.getuid()} -it -v {SYNBOLS_INCLUDE_PATH}:/synbols_include/synbols " + \
          f"{path_mounts} -w {curdir} {DOCKER_IMAGE}:{DOCKER_TAG} " + \
          f"sh -c 'export PYTHONPATH=$PYTHONPATH:/synbols_include; python {file} {args}'"
    os.system(cmd)


def main():
    parser = argparse.ArgumentParser(description="Run a Python script in the Synbols runtime environment.")
    parser.add_argument("file", help="Python script to run in Synbols environment")
    parser.add_argument("--paths", type=str, nargs='+',
                        help="Path to directories other than the local directory to be made accessible at run time")
    args, unknown_args = parser.parse_known_args()

    # Check if Docker is installed
    if not is_docker_installed():
        print("Error: Docker installation not found. Please install Docker before using Synbols. " +
              "See https://docs.docker.com/get-docker/")
        exit(1)

    # Check if the Synbols Docker image is available and pull it if needed
    if not is_docker_image_available():
        print(f"The Synbols Docker image for package version {SYNBOLS_VERSION} is not available on this system. It",
              "will now be downloaded. (This will take a while)")
        os.system("docker pull %s:%s" % (DOCKER_IMAGE, DOCKER_TAG))

    # Check if python script to run exists
    if not os.path.exists(args.file):
        print("Error: The Python script to run (%s) cannot be found." % args.file)
        exit(1)

    run_in_docker(args.file, paths=args.paths, args=unknown_args)


if __name__ == "__main__":
    main()