=================
Quara
=================

|license| |tests| |docs publish| |release| |downloads| |DOI|

.. |license| image:: https://img.shields.io/github/license/tknrsgym/quara
    :alt: license
    :target: https://opensource.org/licenses/Apache-2.0

.. |tests| image:: https://img.shields.io/circleci/build/github/tknrsgym/quara
    :alt: tests
    :target: https://circleci.com/gh/tknrsgym/quara

.. |docs publish| image:: https://readthedocs.org/projects/quara/badge/?version=stable
    :alt: Documentation Status
    :target: https://quara.readthedocs.io/en/stable/

.. |release| image:: https://img.shields.io/github/release/tknrsgym/quara
    :alt: release
    :target: https://github.com/tknrsgym/quara/releases

.. |downloads| image:: https://pepy.tech/badge/quara
    :alt: downloads
    :target: https://pypi.org/project/quara/

.. |DOI| image:: https://zenodo.org/badge/230030298.svg
    :target: https://zenodo.org/badge/latestdoi/230030298

.. _start_of_about:

Quara, which stands for "Quantum Characterization", is an open-source library for characterizing elementary quantum operations. Currently protocols of standard tomography for quantum states, POVMs, and gates are implemented.

.. _end_of_about:

- **Documentation:** https://quara.readthedocs.io/en/stable/
- **Tutorials:** https://github.com/tknrsgym/quara/tree/master/tutorials
- **Source code:** https://github.com/tknrsgym/quara
- **Bug reports:** https://github.com/tknrsgym/quara/issues
- **Contributing:** https://github.com/tknrsgym/quara/blob/master/docs/contributing.rst

.. _start_of_install:

Install
=================================

**Python version:** 3.7+

.. code-block::

   pip install quara

.. _end_of_install:

.. _start_of_install_3rd_packages:


Use with other optimization parsers and solvers
=====================================================

Quara can also be used with other optimization parsers and solvers. The currently supported combinations are as follows:

+---------+---------+------------------------------------------------------------+
| parser  | solver  | install                                                    |
+=========+=========+============================================================+
| CVXPY   |  MOSEK  | See `the CVXPY website <https://www.cvxpy.org/install/>`_  |
+---------+---------+------------------------------------------------------------+


Interface from different packages
==========================================

Quara supports the wrappers for executing standard tomography from several packages. To use this wrapper, install the package to be used as follows:

**Qiskit:**

.. code-block::

   pip install qiskit

**QuTiP:**

.. code-block::

   pip install qutip

**Forest:**

Install Forest SDK, pyQuil, QVM, and Quil Compiler. See `the pyQuil website <https://pyquil-docs.rigetti.com/en/stable/start.html>`_ for installation instructions.


See the tutorial for detailed instructions.

- `Using Quara's standard tomography features from Qiskit <https://github.com/tknrsgym/quara/blob/master/tutorials/tutrial_for_standardtomography_from_qiskit.ipynb>`_
- `Using Quara's standard tomography features from QuTiP <https://github.com/tknrsgym/quara/blob/master/tutorials/tutorial_for_standardqtomography_from_qutip.ipynb>`_
- Using Quara's standard tomography features from Forest [ `1qubit <https://github.com/tknrsgym/quara/blob/master/tutorials/tutorial_for_standardqtomography_from_forest_1qubit.ipynb>`_ / `2qubit <https://github.com/tknrsgym/quara/blob/master/tutorials/tutorial_for_standardqtomography_from_forest_2qubit.ipynb>`_ ]

.. _end_of_install_3rd_packages:

Citation
=================================
If you use Quara in your research, please cite as per the included `CITATION.cff file <https://github.com/tknrsgym/quara/blob/master/CITATION.cff>`_. 


License
=================================

Apache License 2.0 `LICENSE <https://github.com/tknrsgym/quara/blob/master/LICENSE>`_

Supports
=================================

Quara development is supported by `JST, PRESTO Grant Number JPMJPR1915, Japan. <https://www.jst.go.jp/kisoken/presto/en/project/1112090/1112090_2019.html>`_
