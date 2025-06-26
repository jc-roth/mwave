# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mwave'
copyright = '2025, J Roth'
author = 'J Roth'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# extensions = ['autoapi.extension','sphinx_rtd_theme']

extensions = ['sphinx.ext.autodoc','sphinx_rtd_theme', 'nbsphinx']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# # Tell autoapi where to look
# autoapi_dirs=['../src/mwave']
# autoapi_keep_files = False
# autoapi_generate_api_docs = False

# Add the mwave code to the system path
#import sys
#sys.path.append('../src/mwave')

# # Copy the examples folder into docs/code_examples
# import shutil
# import os
# if os.path.exists('code_examples'):
#     shutil.rmtree('code_examples')
# shutil.copytree("../examples", "code_examples", ignore=shutil.ignore_patterns('*.py', '*.ipynb_checkpoints'))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
