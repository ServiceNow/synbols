#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys

import pkg_resources

SYNBOLS_INCLUDE_PATH = os.path.join(
    pkg_resources.require("synbols")[0].location,
    "synbols"
)
SYNBOLS_VERSION = pkg_resources.require("synbols")[0].version
DOCKER_IMAGE = "aldro61/synbols"
DOCKER_TAG = "v%s" % SYNBOLS_VERSION  # The tag matches the package version


def is_docker_installed():
    """Check if Docker is installed

    Returns
    -------
    bool
        True if the docker is installed, False otherwise.
    """
    try:
        subprocess.Popen(["docker"],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT).communicate()
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
    stdout, stderr = subprocess.Popen(["docker",
                                       "images",
                                       "-q",
                                       "%s:%s" % (DOCKER_IMAGE, DOCKER_TAG)],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT).communicate()
    return stdout != ""


def run_in_docker(file, paths, ports, args):
    """
    Run a Python script with the Synbols Docker image

    Parameters
    ----------
    file : str
        The Python script to run
    paths: list of str
        Paths to be made acessible at run time
    ports: list of int
        Ports to be opened at run time
    args : list
        A list of command line arguments to pass to the Python script

    """
    # Get the path to the file to run
    file_path = os.path.dirname(os.path.abspath(file))

    # Generate docker arguments to mount all expected directories
    paths = [] if paths is None else paths
    ports = [] if ports is None else ports
    curdir = os.path.abspath(os.getcwd())
    if "SYNBOLS_DEV_IMAGE" in os.environ:
        docker_image = os.environ["SYNBOLS_DEV_IMAGE"]
        print(f"Developer mode: using custom docker image {docker_image}")
    else:
        docker_image = f"{DOCKER_IMAGE}:{DOCKER_TAG}"

    # Merge all command line arguments
    args = " ".join(args)

    arg_list = f"docker run --rm --user {os.getuid()} -it".split()
    arg_list += ["-v", f"{SYNBOLS_INCLUDE_PATH}:/synbols_include/synbols"]
    for p in paths + [curdir, file_path]:
        arg_list += ["-v", f"{p}:{p}"]
    for p in ports:
        arg_list += ["-p", f"{p}:{p}"]
    arg_list += ["-w", f"{curdir}", docker_image]
    arg_list += ["sh",
                 "-c",
                 "export PYTHONPATH=$PYTHONPATH:/synbols_include;" +
                 f"python {file} {args}"]

    subprocess.run(arg_list)  # might need stderr=sys.stderr, stdout=sys.stdout


def _parse_args_and_print_proper_help(parser):
    args = sys.argv[1:]
    no_help_args = [arg for arg in args if arg not in ['-h', '--help']]
    help_args = [arg for arg in args if arg in ['-h', '--help']]

    if len(help_args) > 0:
        # we need to figure out if we print help of synbols
        # or help of the running script
        if len(no_help_args) == 0:
            # will print help of synbols and exit
            parser.parse_known_args()
        else:
            # will try to extract args for running the script
            try:
                args, script_args = parser.parse_known_args(no_help_args)
                script_args += help_args  # pass the help args to the script
            except SystemExit:
                # will print help of synbols and exit
                print("============")
                parser.parse_known_args()

    else:
        args, script_args = parser.parse_known_args()

    return args, script_args


def main():
    parser = argparse.ArgumentParser(
        description="Run a Python script in the Synbols runtime environment."
    )
    parser.add_argument("file",
                        help="Python script to run in Synbols environment")
    parser.add_argument("--mount-path", type=str, nargs='+',
                        help="Path to directories other than the local " +
                             "directory to be made accessible at run time")
    parser.add_argument("--docker-port", type=int, nargs='+',
                        help="Port to be opened at run time")

    args, script_args = _parse_args_and_print_proper_help(parser)

    # Print configuration info
    print(f"Synbols ({SYNBOLS_VERSION}) found at {SYNBOLS_INCLUDE_PATH}")

    # Check if Docker is installed
    if not is_docker_installed():
        print("Error: Docker installation not found. " +
              "Please install Docker before using Synbols. " +
              "See https://docs.docker.com/get-docker/")
        exit(1)

    # Check if the Synbols Docker image is available and pull it if needed
    if not is_docker_image_available():
        print("The Synbols Docker image for package version" +
              f"{SYNBOLS_VERSION} is not available on this system. It",
              "will now be downloaded. (This will take a while)")
        os.system("docker pull %s:%s" % (DOCKER_IMAGE, DOCKER_TAG))

    # Check if python script to run exists
    if not os.path.exists(args.file):
        print(
            "Error: The Python script to run (%s) cannot be found." % args.file
        )
        exit(1)

    run_in_docker(args.file, paths=args.mount_path,
                  ports=args.docker_port, args=script_args)


if __name__ == "__main__":
    main()
