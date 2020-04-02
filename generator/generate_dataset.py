#!/usr/bin/env python
import argparse
import logging
import subprocess as sp
import sys
import os

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='name of the predefined dataset', default='default')
parser.add_argument('--n_samples', help='number of samples to generate', type=int, default=10000)
parser.add_argument('--no_docker', help="Don't run in docker", action='store_true')


def _docker_run(cmd):
    docker_cmd = ("docker run --user %s" % (os.getuid())).split()
    synbol_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    docker_cmd += ["-v", "%s/generator:/generator" % synbol_dir]
    docker_cmd += ["-v", "%s:/local" % synbol_dir]
    docker_cmd += "synbols sh -c".split()
    docker_cmd.append(cmd)
    sp.call(docker_cmd)


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.no_docker:
        _docker_run("cd /local; python generator/generate_dataset.py --no_docker " + ' '.join(sys.argv[1:]))
    else:

        from synbols.generate import *
        from synbols.data_io import write_npz

        logging.info("Generating %d samples from %s dataset", args.n_samples, args.dataset)

        DATASET_GENERATOR_MAP = {
            'plain': generate_plain_dataset,
            'default': generate_default_dataset,
            'camouflage': generate_camouflage_dataset,
            'segmentation': generate_segmentation_dataset,
        }

        ds_generator = DATASET_GENERATOR_MAP[args.dataset](args.n_samples)

        directory = '%s_n=%d' % (args.dataset, args.n_samples)
        write_npz(directory, ds_generator)
