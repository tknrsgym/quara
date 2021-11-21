import pytest
from itertools import product
from numpy import pi
import numpy.testing as npt

from quara.interface.forest.api import (
    calc_empi_dist_from_observables,
    generate_pauli_strings_from_povm_name,
    calc_coefficient_matrix,
    calc_coefficients,
    generate_pauli_operator_from_pauli_string,
    generate_preprocess_program,
    generate_program_for_1qubit,
)
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.povm_typical import generate_povm_from_name

from pyquil import get_qc, Program
from pyquil.gates import CNOT, H, PHASE
from pyquil.paulis import PauliTerm
from pyquil.experiment import (
    Experiment,
    ExperimentSetting,
    zeros_state,
)
from pyquil.operator_estimation import measure_observables
from quara.objects.qoperation_typical import generate_qoperation

from quara.protocol.qtomography.standard.standard_qst import StandardQst


@pytest.mark.forest
def test_generate_pauli_operator_from_pauli_string():
    # Case 1: check dimension of the generated operator
    pauli_strings = ["Y", "ZY", "XXX", "XIIZ"]
    for pauli_string in pauli_strings:
        dims = [[2 ** len(pauli_string)], [2 ** len(pauli_string)]]
        pauli_operator = generate_pauli_operator_from_pauli_string(pauli_string)
        assert pauli_operator.dims == dims

    # Case 2: raise error when input an invalid pauli string
    invalid_string = "XIV"
    with pytest.raises(ValueError):
        generate_pauli_operator_from_pauli_string(invalid_string)


def _test_generate_pauli_strings_from_povm_name(num):
    # Arrange
    povm_names = ["_".join(i) for i in product("xyz", repeat=num)]

    # Case 1: Length and included characters of pauli strings are correct
    allowed_pauli_chars = set("XYZI")
    for povm_name in povm_names:
        pauli_strings = generate_pauli_strings_from_povm_name(povm_name)
        assert len(pauli_strings) == 2 ** num
        for pauli_string in pauli_strings:
            assert len(pauli_string) == num
            assert set(pauli_string) <= allowed_pauli_chars

    # Case 2: input invalid name
    invalid_names = ["bell", "i_j_k", "some_random_string"]
    for invalid_name in invalid_names:
        with pytest.raises(AssertionError):
            generate_pauli_strings_from_povm_name(invalid_name)


@pytest.mark.forest
@pytest.mark.onequbit
def test_generate_pauli_strings_from_povm_name_1qubit():
    _test_generate_pauli_strings_from_povm_name(num=1)


@pytest.mark.forest
@pytest.mark.twoqubit
def test_generate_pauli_strings_from_povm_name_2qubit():
    _test_generate_pauli_strings_from_povm_name(num=2)


@pytest.mark.forest
@pytest.mark.threequbit
def test_generate_pauli_strings_from_povm_name_3qubit():
    _test_generate_pauli_strings_from_povm_name(num=3)


def _test_calc_coefficients(num, limit=None):
    # Arrange
    povm_names = ["_".join(i) for i in product("xyz", repeat=num)]
    if limit:
        povm_names = povm_names[:limit]
    c_sys = generate_composite_system("qubit", num)
    povms = {}
    pauli_strings = {}
    for povm_name in povm_names:
        povms[povm_name] = generate_povm_from_name(povm_name, c_sys)
        pauli_strings[povm_name] = generate_pauli_strings_from_povm_name(povm_name)

    # Check existence of coefficients
    for povm_name in povm_names:
        for pauli_string in pauli_strings[povm_name]:
            coefficients = calc_coefficients(pauli_string, povms[povm_name])
            coefficients.index(1)
            assert len(coefficients) == 2 ** num


@pytest.mark.forest
@pytest.mark.onequbit
def test_calc_coefficients_1qubit():
    _test_calc_coefficients(1)


@pytest.mark.forest
@pytest.mark.twoqubit
def test_calc_coefficients_2qubit():
    _test_calc_coefficients(2)


@pytest.mark.forest
@pytest.mark.threequbit
def test_calc_coefficients_3qubit():
    _test_calc_coefficients(3)


def _test_calc_coefficient_matrix(num, limit=None):
    # Arrange
    povm_names = ["_".join(i) for i in product("xyz", repeat=num)]
    if limit:
        povm_names = povm_names[:limit]
    c_sys = generate_composite_system("qubit", num)
    povms = {}
    pauli_strings = {}
    for povm_name in povm_names:
        povms[povm_name] = generate_povm_from_name(povm_name, c_sys)
        pauli_strings[povm_name] = generate_pauli_strings_from_povm_name(povm_name)

    # Case 1: check the size of coefficient matrix
    for povm_name in povm_names:
        coefficient_mat = calc_coefficient_matrix(
            pauli_strings[povm_name], povms[povm_name]
        )
        assert coefficient_mat.shape == (2 ** num, 2 ** num)


@pytest.mark.forest
@pytest.mark.onequbit
def test_calc_coefficient_matrix_1qubit():
    _test_calc_coefficient_matrix(num=1)


@pytest.mark.forest
@pytest.mark.twoqubit
def test_calc_coefficient_matrix_2qubit():
    _test_calc_coefficient_matrix(num=2)


@pytest.mark.forest
@pytest.mark.threequbit
def test_calc_coefficient_matrix_3qubit():
    _test_calc_coefficient_matrix(num=3)


def obtain_expectations_for_qst(qc, qubits, program, pauli_strings):
    settings = []
    for pauli_str in pauli_strings:
        out_operator = PauliTerm.from_list(list(zip(pauli_str, qubits)))
        settings.append(ExperimentSetting(zeros_state(qubits), out_operator))
    tomo_experiment = Experiment(settings, program)
    expectations = []
    for pauli_str, res in zip(
        pauli_strings,
        measure_observables(
            qc,
            tomo_experiment,
        ),
    ):
        if res.raw_expectation is None:
            # This is the result for II...I operator
            expectations.append(1.0)
        else:
            expectations.append(res.raw_expectation)
    return expectations


def _test_calc_empi_dist_from_observables(
    true_state, c_sys, qc, num_shots, qubits, program, povm_name
):
    # Arrange observables
    c_sys = generate_composite_system("qubit", len(qubits))
    povm = generate_povm_from_name(povm_name, c_sys)
    pauli_strings = generate_pauli_strings_from_povm_name(povm_name)
    observables = obtain_expectations_for_qst(qc, qubits, program, pauli_strings)
    empi_dist = calc_empi_dist_from_observables(
        observables,
        num_shots,
        pauli_strings,
        povm,
    )

    # Arrange probability distribution
    qst = StandardQst([povm], on_para_eq_constraint=True, schedules="all")
    prob_dists = qst.calc_prob_dists(true_state)

    # Check values
    npt.assert_array_almost_equal(empi_dist[1], prob_dists[0], decimal=2)


@pytest.mark.forest
@pytest.mark.onequbit
@pytest.mark.parametrize(("povm_name"), [("x"), ("y"), ("z")])
def test_calc_empi_dist_from_observables_1qubit(povm_name):
    qc = get_qc("1q-qvm")
    qubits = [0]
    num_shots = 10000
    p = Program()
    p += H(qubits[0])
    p += PHASE(pi / 4, qubits[0])
    p.wrap_in_numshots_loop(num_shots)
    c_sys = generate_composite_system("qubit", len(qubits))
    true_state = generate_qoperation(mode="state", name="a", c_sys=c_sys)
    _test_calc_empi_dist_from_observables(
        true_state, c_sys, qc, num_shots, qubits, p, povm_name
    )


@pytest.mark.forest
@pytest.mark.twoqubit
@pytest.mark.parametrize(("povm_name"), [("x_x"), ("x_y"), ("x_z")])
def test_calc_empi_dist_from_observables_2qubit(povm_name):
    qc = get_qc("2q-qvm")
    qubits = [0, 1]
    num_shots = 10000
    p = Program()
    p += H(qubits[0])
    p += CNOT(qubits[0], qubits[1])
    p.wrap_in_numshots_loop(num_shots)
    c_sys = generate_composite_system("qubit", len(qubits))
    true_state = generate_qoperation(mode="state", name="bell_phi_plus", c_sys=c_sys)
    _test_calc_empi_dist_from_observables(
        true_state, c_sys, qc, num_shots, qubits, p, povm_name
    )


@pytest.mark.forest
@pytest.mark.threequbit
@pytest.mark.parametrize(("povm_name"), [("x_x_x"), ("x_x_y"), ("x_y_z")])
def test_calc_empi_dist_from_observables_3qubit(povm_name):
    qc = get_qc("3q-qvm")
    qubits = [0, 1, 2]
    num_shots = 10000
    p = Program()
    p += H(qubits[0])
    p += CNOT(qubits[0], qubits[1])
    p += CNOT(qubits[1], qubits[2])
    p.wrap_in_numshots_loop(num_shots)
    c_sys = generate_composite_system("qubit", len(qubits))
    true_state = generate_qoperation(mode="state", name="ghz", c_sys=c_sys)
    _test_calc_empi_dist_from_observables(
        true_state, c_sys, qc, num_shots, qubits, p, povm_name
    )


@pytest.mark.forest
def test_generate_program_for_1qubit():
    # Case: generates a Program with correct length when valid input is given
    state_names = ["z0", "z1", "x0", "x1", "y0", "y1"]
    qubits = list(range(10))
    for state_name in state_names:
        for qubit in qubits:
            program = generate_program_for_1qubit(qubit, state_name)
            if state_name == "z0":
                assert len(program.instructions) == 0
            elif state_name == "x1":
                assert len(program.instructions) == 2
            else:
                assert len(program.instructions) == 1

    # Case: invalid input
    invalid_state_names = ["x", "y", "z", "X0", "x0_z0", "u0", "invlid_name"]
    for invalid_state_name in invalid_state_names:
        with pytest.raises(ValueError):
            generate_program_for_1qubit(0, invalid_state_name)


@pytest.mark.forest
def test_generate_preprocess_program():
    # Case: generates a Program when valid input is given
    state_names = ["x0", "x0_y0", "x0_y0_z0"]
    for state_name in state_names:
        qubits = list(range(len(state_name.split("_"))))
        generate_preprocess_program(qubits, state_name)

    # Case: invalid input
    invalid_state_names = ["x", "y", "z", "x_y", "X0_Y0", "x0y0", "u0", "invalid_name"]
    for invalid_state_name in invalid_state_names:
        with pytest.raises(ValueError):
            qubits = list(range(len(invalid_state_name.split("_"))))
            generate_preprocess_program(qubits, invalid_state_name)

    # Case: number of qubits and state name doesn't match
    state_name = "x0_y0_z0"
    qubits = [0, 1]
    with pytest.raises(AssertionError):
        generate_preprocess_program(qubits, state_name)