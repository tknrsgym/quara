# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("../.."))

# for Getting Started
if Path("./getting_started").exists():
    shutil.rmtree("./getting_started")
os.mkdir("./getting_started")
shutil.copy("../README.rst", "./getting_started/README.rst")

# for Jupyter Notebook
if Path("./tutorials").exists():
    shutil.rmtree("./tutorials")
if Path("./tutorials/.ipynb_checkpoints").exists():
    shutil.rmtree("./tutorials/.ipynb_checkpoints")
shutil.copytree("../tutorials", "./tutorials")


# -- Project information -----------------------------------------------------

project = "quara"
copyright = "2020, Tomoko Furuki, Satoyuki Tsukano, Takanori Sugiyama"
author = "Tomoko Furuki, Satoyuki Tsukano, Takanori Sugiyama"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "recommonmark",
]

autoclass_content = "both"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
html_style = "css/my_theme.css"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Generate autosummary pages
autosummary_generate = True

add_module_names = False

master_doc = "index"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    :download:`Download Notebook <https://quara.readthedocs.io/en/latest/{{ docname }}>` / 
    :download:`Download Dataset <https://github.com/tknrsgym/quara/tree/master/tutorials/data>`
"""

nbsphinx_epilog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}

.. only:: html

    :download:`Download Notebook <https://quara.readthedocs.io/en/latest/{{ docname }}>` / 
    :download:`Download Dataset <https://github.com/tknrsgym/quara/tree/master/tutorials/data>`
"""
