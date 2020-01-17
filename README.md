# quara
Quara, which stands for "**Qua**ntum Ch**ara**cterization", is an open-source library for characterizing elemental quantum operations.

## Devlopment

Python version: 3.6+

### Createing environment for deveropment
#### Creating a Python environment

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

#### Install MATLAB engine API for Python

Install MATLAB engine for Python. Refer to [MATLAB documentation](https://jp.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html?lang=en).

### Testing

We currently use [pytest](https://docs.pytest.org/en/latest/) for testing.

To test all the code , enter:

    pytest

To generate a coverage report, enter:

    pytest -v --cov=quara --cov-report=html
    open htmlcov/index.html

### Building the documentation
Refer to ["Building the documentation"](https://github.com/tknrsgym/quara/tree/master/docs)
