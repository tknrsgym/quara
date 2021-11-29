import numpy as np
import numpy.testing as npt
import pytest
from itertools import product

from quara.interface.qulacs.conversion import (
    convert_state_quara_to_qulacs,
    convert_state_qulacs_to_quara,
    convert_gate_quara_to_qulacs,
    convert_instrument_quara_to_qulacs,
)
from quara.objects.composite_system import CompositeSystem
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.operators import compose_qoperations
from quara.objects.state_typical import (
    get_state_names_1qubit,
    get_state_names_2qubit,
    get_state_names_3qubit,
    generate_state_from_name,
)
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.mprocess_typical import generate_mprocess_from_name


def _test_convert_state_between_quara_and_qulacs(mode, num, state_name):
    # Arrange
    c_sys = generate_composite_system(mode, num)
    expected = generate_state_from_name(c_sys, state_name)

    # check that inverse conversion matches to the original data
    source = convert_state_quara_to_qulacs(expected)
    actual = convert_state_qulacs_to_quara(source, c_sys)
    npt.assert_array_almost_equal(actual.vec, expected.vec, decimal=10)


@pytest.mark.qulacs
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_state_names_1qubit()],
)
def test_convert_state_between_quara_and_qulacs_1qubit(state_name):
    _test_convert_state_between_quara_and_qulacs("qubit", 1, state_name)


@pytest.mark.qulacs
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_state_names_2qubit()[:10]],
)
def test_convert_state_between_quara_and_qulacs_2qubit(state_name):
    _test_convert_state_between_quara_and_qulacs("qubit", 2, state_name)


@pytest.mark.qulacs
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name"),
    [(state_name) for state_name in get_state_names_3qubit()[:10]],
)
def test_convert_state_between_quara_and_qulacs_3qubit(state_name):
    _test_convert_state_between_quara_and_qulacs("qubit", 3, state_name)


def _test_convert_gate_between_quara_and_qulacs(
    mode, num, state_name, gate_name, ids=None
):
    c_sys = generate_composite_system(mode, num)
    quara_gate = generate_gate_from_gate_name(gate_name, c_sys, ids)
    quara_state = generate_state_from_name(c_sys, state_name)
    expected = compose_qoperations(quara_gate, quara_state)

    qulacs_state = convert_state_quara_to_qulacs(quara_state)
    qulacs_gate = convert_gate_quara_to_qulacs(quara_gate, list(range(num)))
    qulacs_gate.update_quantum_state(qulacs_state)
    actual = convert_state_qulacs_to_quara(qulacs_state, c_sys)
    npt.assert_array_almost_equal(actual.vec, expected.vec, decimal=10)


@pytest.mark.qulacs
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name", "gate_name"),
    [
        (state_name, gate_name)
        for state_name, gate_name in product(["z0", "x0", "y0"], ["x", "y", "z"])
    ],
)
def test_convert_gate_between_quara_and_qulacs_1qubit(state_name, gate_name):
    _test_convert_gate_between_quara_and_qulacs("qubit", 1, state_name, gate_name)


@pytest.mark.qulacs
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name", "gate_name"),
    [(state_name, gate_name) for state_name, gate_name in product(["z0_x0"], ["zx90"])],
)
def test_convert_gate_between_quara_and_qulacs_2qubit(state_name, gate_name):
    _test_convert_gate_between_quara_and_qulacs(
        "qubit", 2, state_name, gate_name, [0, 1]
    )
    _test_convert_gate_between_quara_and_qulacs(
        "qubit", 2, state_name, gate_name, [1, 0]
    )


@pytest.mark.qulacs
@pytest.mark.threequbit
@pytest.mark.parametrize(
    ("state_name", "gate_name"),
    [
        (state_name, gate_name)
        for state_name, gate_name in product(["z0_z0_z0", "x0_y0_z0"], ["toffoli"])
    ],
)
def test_convert_gate_between_quara_and_qulacs_3qubit(state_name, gate_name):
    _test_convert_gate_between_quara_and_qulacs(
        "qubit", 3, state_name, gate_name, [0, 1, 2]
    )
    _test_convert_gate_between_quara_and_qulacs(
        "qubit", 3, state_name, gate_name, [1, 0, 2]
    )


@pytest.mark.skipci
@pytest.mark.qulacs
@pytest.mark.onequbit
@pytest.mark.parametrize(
    ("state_name", "mprocess_name"),
    [
        (state_name, mprocess_name)
        for state_name, mprocess_name in product(
            ["z0", "x0", "y0"], ["z-type1", "z-type2"]
        )
    ],
)
def test_convert_intrument_between_quara_and_qulacs_1qubit(state_name, mprocess_name):
    c_sys = generate_composite_system("qubit", 1)
    quara_mprocess = generate_mprocess_from_name(c_sys, mprocess_name)
    quara_state = generate_state_from_name(c_sys, state_name)
    expected = compose_qoperations(quara_mprocess, quara_state)

    qulacs_instrument, indices = convert_instrument_quara_to_qulacs(quara_mprocess, [0])

    # Check that nubmers of matrices for each Kraus operator is correct.
    assert indices == [1, 1]

    N = 1000
    counter = 0
    for _ in range(N):
        density_matrix = convert_state_quara_to_qulacs(quara_state)
        qulacs_instrument.update_quantum_state(density_matrix)
        if density_matrix.get_classical_value(0) == 0:
            counter = counter + 1
    simulated_probability_distribution = np.array([counter / N, 1 - counter / N])

    # Check that the probability distribution almost match to each other
    npt.assert_array_almost_equal(
        np.array(expected.prob_dist.ps), simulated_probability_distribution, decimal=1
    )


@pytest.mark.skipci
@pytest.mark.qulacs
@pytest.mark.twoqubit
@pytest.mark.parametrize(
    ("state_name", "mprocess_name"),
    [
        (state_name, mprocess_name)
        for state_name, mprocess_name in product(
            ["z0_z0", "z0_x0", "x0_y0"],
            ["bell-type1", "xxparity-type1", "zzparity-type1"],
        )
    ],
)
def test_convert_intrument_between_quara_and_qulacs_2qubit(state_name, mprocess_name):
    c_sys = generate_composite_system("qubit", 2)
    quara_mprocess = generate_mprocess_from_name(c_sys, mprocess_name)
    quara_state = generate_state_from_name(c_sys, state_name)
    expected = compose_qoperations(quara_mprocess, quara_state)

    qulacs_instrument, indices = convert_instrument_quara_to_qulacs(
        quara_mprocess, [0, 1]
    )

    # Check that nubmers of matrices for each Kraus operator is correct.
    assert len(indices) == quara_mprocess.num_outcomes

    N = 10000
    counter = [0 for _ in range(quara_mprocess.num_outcomes)]
    for _ in range(N):
        density_matrix = convert_state_quara_to_qulacs(quara_state)
        qulacs_instrument.update_quantum_state(density_matrix)
        position = density_matrix.get_classical_value(0)
        counter[position] = counter[position] + 1
    simulated_probability_distribution = np.array([num / N for num in counter])

    # Check that the probability distribution almost match to each other
    npt.assert_array_almost_equal(
        np.array(expected.prob_dist.ps), simulated_probability_distribution, decimal=1
    )
