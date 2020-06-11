# The Synbols dataset

The data generation code relies on multiple system and Python dependencies. We provide a makefile for ease of use.

## Requirements

* Docker

## Building the docker image

```
make docker
```

## Running code

Run the python code in the docker image to make sure all the dependencies are installed. For example:
```
./docker_run.sh generator/generate_dataset.py --dataset=default --n_samples=10000
```

