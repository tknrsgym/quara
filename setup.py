import os

from setuptools import find_packages, setup


def read_file(filename):
    basepath = os.path.dirname(os.path.dirname(__file__))
    filepath = os.path.join(basepath, filename)
    if os.path.exists(filepath):
        return open(filepath).read()
    else:
        return ""


setup(
    name="quara",
    version="0.0a0.dev1",
    author="Tomoko Furuki, Satoyuki Tsukano, Takanori Sugiyama",
    author_email="quara@googlegroups.com",
    packages=find_packages(),
    description='Quara, which stands for "Quantum Characterization", is an open-source library for characterizing elementary quantum operations.',
    long_description=read_file("README.rst"),
    url="https://github.com/tknrsgym/quara",
    license="Apache License 2.0",
    python_requires="~=3.7",
    install_requires=[
        "numpy>=1.18",
        "scipy",
        "pandas",
        "tqdm",
        "plotly",
        "plotly-express",
        "tqdm",
        "xhtml2pdf",
        "psutil",
        "kaleido>=0.1.0,<=0.1.0.post1",
        "jinja2"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Physics",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10",
    ],
    keywords=["quara", "lsqpt", "quantum process tomography"],
    include_package_data=True,
    package_data={'': ['LICENSE']},
)
