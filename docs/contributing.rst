Contributing
============

Creating environment for development
------------------------------------

Creating a Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Python version:** 3.6+

To create environment for development, follow these steps:

Mac OS:

::

   # in the project root directory
   python -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   cd ../
   pip install -e ./quara

Windows:

::

   # in the project root directory
   python -m venv venv
   .\venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements-dev.txt
   cd ../
   pip install -e ./quara

Install MATLAB engine API for Python and other tools
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate the Python environment for development, and follow these steps:

1. `Install MATLAB engine for
   Python <https://github.com/tknrsgym/quara#install-matlab-engine-api-for-python>`__
2. `Install other tools (solvers, parsers,
   etc.) <https://github.com/tknrsgym/quara#install-other-tools-solvers-parsers-etc>`__

Testing
-------

We currently use `pytest <https://docs.pytest.org/en/latest/>`__ for
testing.

To test all the code , enter the following command:

::

   pytest

To generate a coverage report, enter the following command:

::

   pytest -v --cov=quara --cov-report=html
   open htmlcov/index.html

Building the documentation
--------------------------

To learn how to build the documentation, refer to `“Building the
documentation” <https://github.com/tknrsgym/quara/tree/master/docs>`__
