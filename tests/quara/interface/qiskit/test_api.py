from numpy.testing._private.utils import assert_array_almost_equal
from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.interface.qiskit.api import (
    estimate_standard_povmt_from_qiskit,
    estimate_standard_qpt_from_qiskit,
    estimate_standard_qst_from_qiskit,
    generate_empi_dists_from_quara,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst
import numpy as np
import numpy.testing as npt
import pytest

from quara.interface.qiskit.conversion import (
    convert_empi_dists_quara_to_qiskit,
    convert_empi_dists_quara_to_qiskit_shots,
    convert_state_quara_to_qiskit,
    convert_povm_quara_to_qiskit,
    convert_gate_quara_to_qiskit,
)
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.gate_typical import generate_gate_from_gate_name


def get_tester_state_names_1qubit():
    return ["x0", "y0", "z0", "z1"]


def get_tester_povm_names_1qubit():
    return ["x", "y", "z"]


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("mode", "num", "true_state_name", "decimal"), [("qubit", 1, "z0", 4)]
)
def test_estimate_standard_qst_from_qiskit(mode, num, true_state_name, decimal):
    c_sys = generate_composite_system(mode, num)
    true_state = generate_state_from_name(c_sys, true_state_name)
    true_state_qiskit = convert_state_quara_to_qiskit(true_state)

    get_tester_povm_names_method_name = f"get_tester_povm_names_{int(num)}{mode}"
    get_tester_povm_names_method = eval(get_tester_povm_names_method_name)
    get_tester_povm_names = get_tester_povm_names_method()
    tester_povms = []
    tester_povms_qiskit = []
    for tester_povm_name in get_tester_povm_names:
        tester_povm = generate_povm_from_name(tester_povm_name, c_sys)
        tester_povms.append(tester_povm)
        tester_povms_qiskit.append(convert_povm_quara_to_qiskit(tester_povm))

    seed = 7896
    qst = StandardQst(
        tester_povms, on_para_eq_constraint=True, schedules="all", seed_data=seed
    )
    prob_dists_arrays = qst.calc_prob_dists(true_state)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    empi_dists_qiskit = convert_empi_dists_quara_to_qiskit(prob_dists)
    shots = convert_empi_dists_quara_to_qiskit_shots(prob_dists)
    label = [2, 2, 2]

    for estimator_name in ["linear", "least_squares"]:
        estimated_state_qiskit = estimate_standard_qst_from_qiskit(
            mode,
            num,
            tester_povms=tester_povms_qiskit,
            empi_dists=empi_dists_qiskit,
            shots=shots,
            label=label,
            estimator_name=estimator_name,
            schedules="all",
        )
        npt.assert_array_almost_equal(
            estimated_state_qiskit,
            true_state_qiskit,
            decimal=decimal,
        )


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("mode", "num", "true_povm_name", "decimal"), [("qubit", 1, "z", 4)]
)
def test_estimate_standard_povmt_from_qiskit(mode, num, true_povm_name, decimal):
    c_sys = generate_composite_system(mode, num)
    true_povm = generate_povm_from_name(true_povm_name, c_sys)
    true_povm_qiskit = convert_povm_quara_to_qiskit(true_povm)

    get_tester_state_names_method_name = f"get_tester_state_names_{int(num)}{mode}"
    get_tester_state_names_method = eval(get_tester_state_names_method_name)
    get_tester_state_names = get_tester_state_names_method()
    tester_states = []
    tester_states_qiskit = []
    for tester_state_name in get_tester_state_names:
        tester_state = generate_state_from_name(c_sys, tester_state_name)
        tester_states.append(tester_state)
        tester_states_qiskit.append(convert_state_quara_to_qiskit(tester_state))

    seed = 7896
    povmt = StandardPovmt(
        tester_states,
        true_povm.num_outcomes,
        on_para_eq_constraint=True,
        schedules="all",
        seed_data=seed,
    )
    prob_dists_arrays = povmt.calc_prob_dists(true_povm)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    empi_dists_qiskit = convert_empi_dists_quara_to_qiskit(prob_dists)
    shots = convert_empi_dists_quara_to_qiskit_shots(prob_dists)
    label = [2, 2, 2, 2]

    for estimator_name in ["linear", "least_squares"]:
        estimated_povm_qiskit = estimate_standard_povmt_from_qiskit(
            mode,
            num,
            tester_states=tester_states_qiskit,
            empi_dists=empi_dists_qiskit,
            shots=shots,
            label=label,
            num_outcomes=true_povm.num_outcomes,
            estimator_name=estimator_name,
            schedules="all",
        )
        npt.assert_array_almost_equal(
            estimated_povm_qiskit,
            true_povm_qiskit,
            decimal=decimal,
        )


@pytest.mark.qiskit
@pytest.mark.parametrize(
    ("mode", "num", "true_gate_name", "decimal"), [("qubit", 1, "identity", 4)]
)
def test_estimate_standard_qpt_from_qiskit(mode, num, true_gate_name, decimal):
    dim = 2 ** num
    c_sys = generate_composite_system(mode, num)
    true_gate = generate_gate_from_gate_name(true_gate_name, c_sys)
    true_gate_qiskit = convert_gate_quara_to_qiskit(true_gate, dim)

    get_tester_state_names_method_name = f"get_tester_state_names_{int(num)}{mode}"
    get_tester_state_names_method = eval(get_tester_state_names_method_name)
    get_tester_state_names = get_tester_state_names_method()
    tester_states = []
    tester_states_qiskit = []
    for tester_state_name in get_tester_state_names:
        tester_state = generate_state_from_name(c_sys, tester_state_name)
        tester_states.append(tester_state)
        tester_states_qiskit.append(convert_state_quara_to_qiskit(tester_state))

    get_tester_povm_names_method_name = f"get_tester_povm_names_{int(num)}{mode}"
    get_tester_povm_names_method = eval(get_tester_povm_names_method_name)
    get_tester_povm_names = get_tester_povm_names_method()
    tester_povms = []
    tester_povms_qiskit = []
    for tester_povm_name in get_tester_povm_names:
        tester_povm = generate_povm_from_name(tester_povm_name, c_sys)
        tester_povms.append(tester_povm)
        tester_povms_qiskit.append(convert_povm_quara_to_qiskit(tester_povm))

    seed = 7896
    qpt = StandardQpt(
        tester_states,
        tester_povms,
        on_para_eq_constraint=True,
        schedules="all",
        seed_data=seed,
    )
    prob_dists_arrays = qpt.calc_prob_dists(true_gate)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    empi_dists_qiskit = convert_empi_dists_quara_to_qiskit(prob_dists)
    shots = convert_empi_dists_quara_to_qiskit_shots(prob_dists)
    label = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    for estimator_name in ["linear", "least_squares"]:
        estimated_gate_qiskit = estimate_standard_qpt_from_qiskit(
            mode,
            num,
            tester_states=tester_states_qiskit,
            tester_povms=tester_povms_qiskit,
            empi_dists=empi_dists_qiskit,
            shots=shots,
            label=label,
            estimator_name=estimator_name,
            schedules="all",
        )
        npt.assert_array_almost_equal(
            estimated_gate_qiskit,
            true_gate_qiskit,
            decimal=decimal,
        )


@pytest.mark.qiskit
def test_generate_empi_dists_from_quara_label():
    true_empi_dists = [[2, 2, 2], np.array([0.864, 0.136, 0.844, 0.156, 0.49, 0.51])]
    source = [
        (1000, np.array([0.864, 0.136])),
        (1000, np.array([0.844, 0.156])),
        (1000, np.array([0.49, 0.51])),
    ]
    actual = generate_empi_dists_from_quara(source)
    assert actual[0] == true_empi_dists[0]


@pytest.mark.qiskit
def test_generate_empi_dists_from_quara_dists():
    true_empi_dists = [[2, 2, 2], np.array([0.864, 0.136, 0.844, 0.156, 0.49, 0.51])]
    source = [
        (1000, np.array([0.864, 0.136])),
        (1000, np.array([0.844, 0.156])),
        (1000, np.array([0.49, 0.51])),
    ]
    actual = generate_empi_dists_from_quara(source)
    npt.assert_array_almost_equal(true_empi_dists[1], actual[1])
