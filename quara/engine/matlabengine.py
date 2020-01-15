import os
from pathlib import Path

import matlab


class MatlabEngine(object):
    def __init__(self):
        this_pypath = Path(os.path.abspath(__file__))
        self._matlab_func_path = this_pypath.parent.parent.parent / "matlab"

    def __enter__(self):
        self.engine = matlab.engine.start_matlab()

        self.engine.addpath(str(self._matlab_func_path), nargout=0)

        for path in self._matlab_func_path.glob("**"):
            self.engine.addpath(str(path))
        # print(self.engine.path())
        # debug
        # self.engine.addpath("/Users/furuki/applications/sedumi-master")
        # self.engine.addpath("/Users/furuki/applications/yalmip")
        self.engine.addpath("/Users/furuki/Applications/YALMIP/@sdpvar/")
        self.engine.addpath("/Applications/MATLAB_R2019b.app/toolbox/matlab/sparfun/")

        return self.engine

    def __exit__(self, exc_type, exc_value, traceback):
        self.engine.quit()
