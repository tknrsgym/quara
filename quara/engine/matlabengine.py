import os
from pathlib import Path

import matlab.engine


class MatlabEngine(object):
    """Wrapper class for safe use of Matlab.

    Attributes
    ----------
    _matlab_func_path : pathlib.Path
        Matlab's function path.
    _engine : matlab.engine.matlabengine.MatlabEngine
        Matlab object.

    Examples
    --------
    Follow this manner when calling Matlab from Python:

    >>> with MatlabEngine() as engine:
    >>>     engine.matlab_function()
    """

    def __init__(self):
        this_pypath = Path(os.path.abspath(__file__))
        self._matlab_func_path = this_pypath.parent.parent / "matlab_script"

    def __enter__(self):
        self._engine = matlab.engine.start_matlab()

        self._engine.cd(str(self._matlab_func_path), nargout=0)
        for path in self._matlab_func_path.glob("**"):
            self._engine.addpath(str(path))

        return self._engine

    def __exit__(self, exc_type, exc_value, traceback):
        self._engine.quit()
