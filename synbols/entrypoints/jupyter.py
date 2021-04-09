#!/usr/bin/env python
"""Launch a Jupyter notebook in the Synbols runtime environment"""
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main():
    args = parse_args()
    port = args.port
    display = "--NotebookApp.custom_display_url=http://0.0.0.0:{}".format(port)
    subprocess.call(
        [
            "jupyter",
            "notebook",
            "--allow-root",
            "--ip=0.0.0.0",
            "--port=%d" % args.port,
            "--no-browser",
            "--NotebookApp.token=",
            "--NotebookApp.disable_check_xsrf=True",
            "--NotebookApp.allow_origin=*",
            display,
        ]
    )


def entrypoint():
    args = parse_args()
    port = str(args.port)
    subprocess.call(["synbols", __file__, "--docker-port", port, "--port", port])


if __name__ == "__main__":
    main()
