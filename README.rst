=================
quara
=================

.. _start_of_about:

Quara, which stands for "Quantum Characterization", is an open-source library for characterizing elemental quantum operations.

.. _end_of_about:

- **Documentation:** https://quara.readthedocs.io/en/latest/
- **Tutorials:** https://quara.readthedocs.io/en/latest/index.html#tutorials
- **Source code:** https://github.com/quara/quara
- **Bug reports:** https://github.com/quara/quara/issues
- **Contributing:** https://quara.readthedocs.io/en/latest/contributing.html

.. _start_of_install:

Install
=================================

**Python version:** 3.6+

To use Quara, follow these steps:

1. Install Quara
2. Install MATLAB engine API for Python
3. Install other tools (solvers, parsers, etc.)

1. Install Quara
----------------------

.. code-block::

   pip install quara

2. Install MATLAB engine API for Python
--------------------------------------------
The algorithms in Quara are implemented in MATLAB. To use Quara, install the MATLAB engine API for Python.
The MATLAB engine API is not provided on PyPI. To learn how to install, refer to `MATLAB documentation <https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html?lang=en>`_ .


3. Install other tools (solvers, parsers, etc.)
------------------------------------------------------------------
We currently use `SeDuMi <http://sedumi.ie.lehigh.edu/>`_ as a solver for semidefinite programming and `YALMIP <https://yalmip.github.io/>`_ as a parser. To use Quara, install these.

.. _end_of_install:
