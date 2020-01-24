# How to build the docs

We currently use [Sphinx](https://www.sphinx-doc.org/en/master/) for generating documentation for quara.

## Requirements

First, you neeed to have a environment for developing quara (see the [docs on creating a development environment](https://github.com/tknrsgym/quara#devlopment))

Next, install Pandoc (see the [Pandoc documents](https://pandoc.org/installing.html)).


## Building the docs

To build the HTML documentation, enter the following command in the ``docs/`` directory:

    source build_docs.sh

Then you can find the HTML output in the directory ``quara/docs/_build/html/``.
