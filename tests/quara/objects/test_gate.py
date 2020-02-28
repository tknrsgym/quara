import numpy as np
import numpy.testing as npt

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import (
    Gate,
    get_i,
    get_x,
    get_y,
    get_z,
    get_h,
    get_root_x,
    get_root_y,
    get_s,
    get_sdg,
    get_t,
)


"""
def test_convert_to_comp_basis():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    gate = get_h(c_sys)
    print(gate.convert_to_comp_basis())
    # TODO implement
"""


def test_get_i():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(x.hs @ i.hs, x.hs, decimal=15)
    npt.assert_almost_equal(i.hs @ x.hs, x.hs, decimal=15)
    npt.assert_almost_equal(y.hs @ i.hs, y.hs, decimal=15)
    npt.assert_almost_equal(i.hs @ y.hs, y.hs, decimal=15)
    npt.assert_almost_equal(z.hs @ i.hs, z.hs, decimal=15)
    npt.assert_almost_equal(i.hs @ z.hs, z.hs, decimal=15)


def test_get_x():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(x.hs @ x.hs, i.hs, decimal=15)
    npt.assert_almost_equal(x.hs @ y.hs, z.hs, decimal=15)


def test_get_y():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(y.hs @ y.hs, i.hs, decimal=15)
    npt.assert_almost_equal(y.hs @ z.hs, x.hs, decimal=15)


def test_get_z():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    i = get_i(c_sys)
    x = get_x(c_sys)
    y = get_y(c_sys)
    z = get_z(c_sys)
    npt.assert_almost_equal(z.hs @ z.hs, i.hs, decimal=15)
    npt.assert_almost_equal(z.hs @ x.hs, y.hs, decimal=15)


def test_get_root_x():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    x = get_x(c_sys)
    rootx = get_root_x(c_sys)
    npt.assert_almost_equal(rootx.hs @ rootx.hs, x.hs, decimal=15)


def test_get_root_y():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    y = get_y(c_sys)
    rooty = get_root_y(c_sys)
    npt.assert_almost_equal(rooty.hs @ rooty.hs, y.hs, decimal=10)


def test_get_s():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    z = get_z(c_sys)
    s = get_s(c_sys)
    npt.assert_almost_equal(s.hs @ s.hs, z.hs, decimal=15)


def test_get_sdg():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    s = get_s(c_sys)
    sdg = get_sdg(c_sys)
    i = get_i(c_sys)
    npt.assert_almost_equal(s.hs @ sdg.hs, i.hs, decimal=15)


def test_get_t():
    e_sys = ElementalSystem("q0", matrix_basis.get_normalized_pauli_basis())
    c_sys = CompositeSystem([e_sys])
    s = get_s(c_sys)
    t = get_t(c_sys)
    npt.assert_almost_equal(t.hs @ t.hs, s.hs, decimal=15)
