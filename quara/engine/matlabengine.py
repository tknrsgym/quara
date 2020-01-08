import os
from pathlib import Path

import matlab


class MatlabEngine(object):
    def __init__(self):
        this_pypath = Path(os.path.abspath(__file__))
        # TODO: matlab関数を配置するフォルダについてはmtgで検討する
        self._matlab_func_path = str(this_pypath.parent.parent / "core" / "matlab")

    def __enter__(self):
        self.eng = matlab.engine.start_matlab()
        self.eng.cd(self._matlab_func_path, nargout=0)
        return self.eng

    def __exit__(self, exc_type, exc_value, traceback):
        self.eng.quit()
