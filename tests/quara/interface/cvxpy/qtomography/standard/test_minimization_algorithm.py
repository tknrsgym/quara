import numpy as np
import numpy.testing as npt
import pytest

from cvxpy.expressions.variable import Variable as CvxpyVariable
from cvxpy.expressions.expression import Expression as CvxpyExpression

from quara.interface.cvxpy.qtomography.standard import minimization_algorithm


class TestCvxpyMinimizationResult:
    def test_access_variable_value(self):
        pass
