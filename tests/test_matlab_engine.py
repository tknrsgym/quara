import os

import matlab
import numpy as np
import pytest

import matlab.engine

print("Start!")
eng = matlab.engine.start_matlab()
# str_version = eng.version()
# print(str_version)
list_a = matlab.double([1, 2, 3])
# list_b = [-1, -2, -3]
# a = matlab.int8([1])
eng.check_pass_from_python_to_matlab(list_a, nargout=0)
# b = eng.sqrt(list_a)
# print(b)
eng.quit()

print("End!")
