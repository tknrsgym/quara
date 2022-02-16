import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.povm_typical import generate_povm_from_name
from quara.protocol.qtomography.standard.preprocessing import (
    StandardQTomographyPreprocessing,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst


def get_test_data_qst(on_para_eq_constraint=True):
    c_sys = generate_composite_system("qubit", 1)

    povm_x = generate_povm_from_name("x", c_sys)
    povm_y = generate_povm_from_name("y", c_sys)
    povm_z = generate_povm_from_name("z", c_sys)
    povms = [povm_x, povm_y, povm_z]

    qst = StandardQst(povms, on_para_eq_constraint=on_para_eq_constraint, seed_data=7)

    return qst, c_sys


def test_init():
    sqt, c_sys = get_test_data_qst()
    StandardQTomographyPreprocessing(sqt)
