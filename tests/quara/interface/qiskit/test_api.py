from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.interface.qiskit.api import (
    estimate_standard_povmt_from_qiskit,
    estimate_standard_qpt_from_qiskit,
    estimate_standard_qst_from_qiskit,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst
import numpy as np
import numpy.testing as npt
import pytest

from quara.interface.qiskit.conversion import (
    convert_empi_dists_quara_to_qiskit,
    convert_empi_dists_quara_to_qiskit_shots,
    convert_povm_qiskit_to_quara,
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


def get_tester_povms_names_1qubit():
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
        tester_povms_qiskit.append(convert_povm_qiskit_to_quara(tester_povm, c_sys))

    seed = 7896
    qst = StandardQst(
        tester_povms, on_para_eq_constraint=True, schedules="all", seed=seed
    )
    prob_dists_arrays = qst.calc_prob_dists(true_state)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    empi_dists_quara = convert_empi_dists_quara_to_qiskit(prob_dists)
    shots = convert_empi_dists_quara_to_qiskit_shots(prob_dists)
    label = [2, 2, 2]

    for estimator_name in ["linear", "least_squares"]:
        estimated_state_qiskit = estimate_standard_qst_from_qiskit(
            mode,
            num,
            tester_povms=tester_povms_qiskit,
            empi_dists=empi_dists_quara,
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
@pytest.mark.parametrize