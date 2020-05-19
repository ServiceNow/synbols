#!/bin/bash

synbol_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
docker run --user $(id -u) -it \
    -v $synbol_dir:/local \
    -w /local \
    synbols \
    python "$@"
