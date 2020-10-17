from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    # Package description
    name="synbols",
    version="0.1.3.dev",  # XXX: developers, if you change the version, please tag and push the docker image (see doc)
    author='Alexandre Lacoste, Pau Rodriguez, Frederic Branchaud-Charron, Parmida Atighehchian, Massimo Caccia, ' +
           'Issam Hadj Laradji, Alexandre Drouin, Matt Craddock, Laurent Charlin, David Vazquez',
    author_email='allac@elementai.com',
    description='Synbols: Probing Learning Algorithms with Synthetic Datasets',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/ElementAI/synbols',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],

    # Data files required at runtime
    packages=find_packages(),
    package_data={
        'synbols.fonts.blacklist': ['blacklist_english.tsv', 'font_clusters_english.json'],
    },

    # Install the Synbols runner
    entry_points={
        'console_scripts': ['synbols=synbols.entrypoints.run_docker:main',
                            'synbols-datasets=synbols.entrypoints.generate_datasets:entrypoint',
                            'synbols-view=synbols.entrypoints.view_dataset:entrypoint',
                            'synbols-jupyter=synbols.entrypoints.jupyter:entrypoint'],
    },

    # Dependencies
    install_requires=["h5py"],
    python_requires='>=3.6'
)
