# quara
Quara, which stands for "**Qua**ntum Ch**ara**cterization", is an open-source library for characterizing elemental quantum operations.

## Install

**Python version:** 3.6+

To use Quara, follow these steps:

1. [Install Quara](https://github.com/tknrsgym/quara#install-quara)
2. [Install MATLAB engine API for Python](https://github.com/tknrsgym/quara#install-matlab-engine-api-for-python)
3. [Install other tools (solvers, parsers, etc.)](https://github.com/tknrsgym/quara#install-matlab-engine-api-for-python)

### Install Quara

```
pip install quara
```

### Install MATLAB engine API for Python
The algorithms in Quara are implemented in MATLAB. To use Quara, install the MATLAB engine API for Python.
The MATLAB engine API is not provided on PyPI. To learn how to install, refer to [MATLAB documentation](https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html?lang=en).

### Install other tools (solvers, parsers, etc.)
We currently use [SeDuMi](http://sedumi.ie.lehigh.edu/) as a solver for semidefinite programming and [YALMIP](https://yalmip.github.io/) as a parser. To use Quara, install these.

## Devlopment

### Createing environment for deveropment
#### Creating a Python environment

**Python version:** 3.6+

Mac OS:
```
# in the project root directory
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cd ../
pip install -e ./quara
```

Windows:
```
# in the project root directory
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-dev.txt
cd ../
pip install -e ./quara
```

#### Install MATLAB engine API for Python and other tools

- [Install MATLAB engine for Python](https://github.com/tknrsgym/quara#install-matlab-engine-api-for-python)
- [Install other tools (solvers, parsers, etc.)](https://github.com/tknrsgym/quara#install-other-tools-solvers-parsers-etc)

### Testing

We currently use [pytest](https://docs.pytest.org/en/latest/) for testing.

To test all the code , enter the following command:

    pytest

To generate a coverage report, enter the following command:

    pytest -v --cov=quara --cov-report=html
    open htmlcov/index.html

### Building the documentation
To learn how to build the documentation, refer to ["Building the documentation"](https://github.com/tknrsgym/quara/tree/master/docs)
