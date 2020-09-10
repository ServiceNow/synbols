.. Synbols documentation master file, created by
   sphinx-quickstart on Thu Aug  6 01:59:19 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the Synbols documentation!
================================================

.. image:: https://github.com/ElementAI/synbols/raw/master/cover.png
           :alt: Synbols Cover
   
Progress in the field of machine learning has been fueled by the introduction of benchmark datasets pushing the limits of existing algorithms.  Enabling the design of datasets to test specific properties and failure modes of learning algorithms is thus a problem of high interest, as it has a direct impact on innovation in the field. In this sense, we introduce Synbols — Synthetic Symbols — a tool for rapidly generating new datasets with a rich composition of latent features rendered in low resolution images. Synbols leverages the large amount of symbols available in the Unicode standard and the wide range of artistic font provided by the open font community. Our tool's high-level interface provides a language for rapidly generating new distributions on the latent features, including various types of textures and occlusions. To showcase the versatility of Synbols, we use it to dissect the limitations and flaws in standard learning algorithms in various learning setups including supervised learning, active learning, out of distribution generalization, unsupervised representation learning, and object counting.

Installation
************

The easiest way to install Synbols is via `PyPI <https://pypi.org/project/synbols/>`_. Simply run the following command:

.. code-block:: bash

   pip install synbols


Software dependencies
*********************

Synbols relies on fonts and system packages. To ensure reproducibility, we provide a `Docker image <https://hub.docker.com/repository/docker/aldro61/synbols>`_ with everything
preinstalled. Thus, the only dependency is `Docker <https://docs.docker.com/get-docker/>`_.

Usage
*****

To run your code in the Synbols runtime environment, simply use the `synbols` command as follows:

.. code-block:: bash
                
   synbols mydataset.py --foo bar


Contact
*******

For any bug or feature requests, please create an issue.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
