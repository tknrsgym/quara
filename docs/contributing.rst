Contributing
============

Creating environment for development
------------------------------------

Creating a Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Python version:** 3.7+

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

Testing
-------

We currently use `pytest <https://docs.pytest.org/en/latest/>`__ for
testing.

To test all the code, enter the following command:

::

   pytest

To generate a coverage report, enter the following command:

::

   pytest -v --cov=quara --cov-report=html
   open htmlcov/index.html

Building the documentation
--------------------------

To learn how to build the documentation, refer to `"How to build the docs" <https://github.com/tknrsgym/quara/tree/master/docs#how-to-build-the-docs>`__
