# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Synbols'
copyright = '2020, Alexandre Lacoste ( Element AI) < allac@elementai.com>, Pau Rodriguez ( Element AI) < pau.rodriguez@elementai.com>, Frederic Branchaud-Charron ( Element AI) < frederic.branchaud-charron@elementai.com>, Parmida Atighehchian ( Element AI) < parmida@elementai.com>, Massimo Caccia ( MILA) < massimo.p.caccia@gmail.com>, Issam Hadj Laradji ( Element AI) < issam.laradji@gmail.com>, Alexandre Drouin ( Element AI) < adrouin@elementai.com>, Matt Craddock ( Element AI) < matt.craddock@elementai.com>, Laurent Charlin ( HEC Montreal and Mila) < lcharlin@gmail.com>, David Vazquez ( Element AI) < dvazquez@elementai.com>'
author = 'Alexandre Lacoste ( Element AI) < allac@elementai.com>, Pau Rodriguez ( Element AI) < pau.rodriguez@elementai.com>, Frederic Branchaud-Charron ( Element AI) < frederic.branchaud-charron@elementai.com>, Parmida Atighehchian ( Element AI) < parmida@elementai.com>, Massimo Caccia ( MILA) < massimo.p.caccia@gmail.com>, Issam Hadj Laradji ( Element AI) < issam.laradji@gmail.com>, Alexandre Drouin ( Element AI) < adrouin@elementai.com>, Matt Craddock ( Element AI) < matt.craddock@elementai.com>, Laurent Charlin ( HEC Montreal and Mila) < lcharlin@gmail.com>, David Vazquez ( Element AI) < dvazquez@elementai.com>'


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinxcontrib.napoleon']
apidoc_module_dir = '../synbols'
apidoc_output_dir = 'synbols'
apidoc_excluded_paths = []
apidoc_separate_modules = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Borrowed from https://github.com/readthedocs/readthedocs.org/issues/1139
def run_apidoc(_):
    from sphinx.ext.apidoc import main
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    module = 'synbols'
    output_path = os.path.join(cur_dir, 'synbols')
    main(['-o', output_path, module])

def setup(app):
    app.connect('builder-inited', run_apidoc)
