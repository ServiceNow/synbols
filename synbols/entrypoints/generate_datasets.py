#!/usr/bin/env python
import argparse
import logging
import subprocess
import sys
from synbols.utils import language_map_statistics
from argparse import RawDescriptionHelpFormatter

from datetime import datetime


def main():
    # XXX: Imports are here so that they are done
    # inside the docker image (synbols [...])
    from synbols.generate import make_preview
    from synbols.predefined_datasets import DATASET_GENERATOR_MAP
    from synbols.data_io import write_h5
    from synbols.fonts import LANGUAGE_MAP

    logging.basicConfig(level=logging.INFO)

    epilog = (
        "Details of symbols available for each languages. More or less can be made available depending on "
        + "arguemtns to get_alphabet(): \n%s" % language_map_statistics()
    )

    parser = argparse.ArgumentParser(epilog=epilog, formatter_class=RawDescriptionHelpFormatter)

    dataset_names = " | ".join(DATASET_GENERATOR_MAP.keys())
    language_names = " | ".join(LANGUAGE_MAP.keys())

    parser.add_argument(
        "--dataset", help="Name of the predefined dataset. One of %s" % dataset_names, default="default"
    )
    parser.add_argument("--n_samples", help="number of samples to generate", type=int, default=10000)
    parser.add_argument(
        "--language", help="Which language's alphabet to use. One of %s" % language_names, default="default"
    )
    parser.add_argument(
        "--resolution", help="""Image resolution e.g.: "32x32". Defaults to the dataset's default.""", default="default"
    )
    parser.add_argument(
        "--seed", help="""The seed of the random number generator. Defaults to None.""", type=int, default=None
    )

    args = parser.parse_args()

    if args.language == "default":
        language = "english"
        file_path = "%s_n=%d_%s" % (args.dataset, args.n_samples, datetime.now().strftime("%Y-%b-%d"))
    else:
        language = args.language
        file_path = "%s(%s)_n=%d_%s" % (args.dataset, language, args.n_samples, datetime.now().strftime("%Y-%b-%d"))

    dataset_function = DATASET_GENERATOR_MAP[args.dataset]

    print("Generating %s dataset. Info: %s" % (args.dataset, dataset_function.__doc__))
    ds_generator = dataset_function(args.n_samples, language=language, seed=args.seed)
    ds_generator = make_preview(ds_generator, file_path + "_preview.png")
    write_h5(file_path + ".h5py", ds_generator, args.n_samples)


def entrypoint():
    subprocess.call(["synbols", __file__] + sys.argv[1:])


if __name__ == "__main__":
    main()
