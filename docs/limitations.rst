===========
Limitations
===========

Current version of Quara supports several estimators for standard quantum tomography of state, POVM, and gate on multi-qubit and multi-qutrit systems. However, its own numerical optimization solver is quite slow, and we do not recommend to perform tomographic data-processing with the current version of Quara for systems larger than 1-qubit. Feasibility on physicality constrains of the solver is also not enough in the current version. Improvements on the optimization speed and constraint feasibility are necessary for Quara.
