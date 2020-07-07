# Synbols: Probing Learning Algorithms with Synthetic Datasets

![Synbols](https://github.com/ElementAI/synbols/raw/local_and_docker/cover.png)

> Progress in the field of machine learning has been fueled by the introduction of benchmark datasets pushing the limits of existing algorithms.  Enabling the design of datasets to test specific properties and failure modes of learning algorithms is thus a problem of high interest, as it has a direct impact on innovation in the field. In this sense, we introduce Synbols — Synthetic Symbols — a tool for rapidly generating new datasets with a rich composition of latent features rendered in low resolution images. Synbols leverages the large amount of symbols available in the Unicode standard and the wide range of artistic font provided by the open font community. Our tool's high-level interface provides a language for rapidly generating new distributions on the latent features, including various types of textures and occlusions. To showcase the versatility of Synbols, we use it to dissect the limitations and flaws in standard learning algorithms in various learning setups including supervised learning, active learning, out of distribution generalization, unsupervised representation learning, and object counting.

[[paper]](link)

## Installation

The easiest way to install Synbols is via [PyPI](https://pypi.org/project/synbols/). Simply run the following command:

`pip install synbols`


## Software dependencies

Synbols relies on fonts and system packages. To ensure reproducibility, we provide a [Docker image](https://hub.docker.com/repository/docker/aldro61/synbols) with everything
preinstalled. Thus, the only dependency is [Docker](https://docs.docker.com/get-docker/).

## Usage

To run you code in the Synbols runtime environment, simply use the `synbols` command as follows:

`synbols mydataset.py --foo bar`


## Contact

For any bug or feature requests, please create an issue.
