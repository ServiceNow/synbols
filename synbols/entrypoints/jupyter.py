#!/usr/bin/env python
"""Launch a Jupyter notebook in the Synbols runtime environment"""
import argparse
import numpy as np
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main():
    args = parse_args()
    subprocess.call(["jupyter", "notebook", "--allow-root", "--ip=0.0.0.0", "--port=%d" % args.port, "--no-browser",
                     "--NotebookApp.token=", "--NotebookApp.disable_check_xsrf=True", "--NotebookApp.allow_origin=*",
                     "--NotebookApp.custom_display_url=http://0.0.0.0:%d" % args.port])

def entrypoint():
    # TODO: add a port parameter to the synbols command to map the image port to the local machine
    #       subprocess.call(["synbols", "--port %d" % args.port, __file__])
    args = parse_args()
    subprocess.call(["synbols", __file__] + sys.argv[1:])


if __name__ == "__main__":
    main()
