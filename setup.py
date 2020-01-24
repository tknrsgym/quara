from setuptools import setup, find_packages

setup(
    name="quara",
    version="0.0.1",
    author="Takanori Sugiyama",
    author_email="",  # TODO
    packages=find_packages(),
    description='Quara, which stands for "Quantum Characterization", is an open-source library for characterizing elemental quantum operations.',
    url="https://github.com/tknrsgym/quara",  # TODO: update
    python_requires="~=3.6",
    install_requires=["numpy>=1.18"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
    ],
    keywords=["quara"],  # TODO
    package_data={"quara.matlab": ["checker/*.m"]},
)

