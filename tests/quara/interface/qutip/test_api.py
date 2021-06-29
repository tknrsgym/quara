from quara.protocol.qtomography.standard.standard_qpt import StandardQpt
from quara.protocol.qtomography.standard.standard_povmt import StandardPovmt
from quara.interface.qutip.api import (
    estimate_standard_povmt_from_qutip,
    estimate_standard_qpt_from_qutip,
    estimate_standard_qst_from_qutip,
)
from quara.protocol.qtomography.standard.standard_qst import StandardQst
import numpy as np
import numpy.testing as npt
import pytest

from quara.interface.qutip.conversion import (
    convert_state_quara_to_qutip,
    convert_povm_quara_to_qutip,
    convert_gate_quara_to_qutip,
)
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.state_typical import generate_state_from_name
from quara.objects.povm_typical import generate_povm_from_name
from quara.objects.gate_typical import generate_gate_from_gate_name


def get_tester_state_names_1qubit():
    return ["x0", "y0", "z0", "z1"]


def get_tester_state_names_1qutrit():
    return [
        "01z0",
        "12z0",
        "02z1",
        "01x0",
        "01y0",
        "12x0",
        "12y0",
        "02x0",
        "02y0",
    ]


def get_tester_povm_names_1qubit():
    return ["x", "y", "z"]


def get_tester_povm_names_1qutrit():
    return ["01x3", "01y3", "z3", "12x3", "12y3", "02x3", "02y3"]


@pytest.mark.qutip
@pytest.mark.parametrize(
    ("mode", "num", "true_state_name", "decimal"),
    [("qubit", 1, "z0", 4), ("qutrit", 1, "01z0", 4)],
)
def test_estimate_standard_qst_from_qutip(mode, num, true_state_name, decimal):
    c_sys = generate_composite_system(mode, num)
    true_state = generate_state_from_name(c_sys, true_state_name)
    true_state_qutip = convert_state_quara_to_qutip(true_state)

    get_tester_povm_names_method_name = f"get_tester_povm_names_{int(num)}{mode}"
    get_tester_povm_names_method = eval(get_tester_povm_names_method_name)
    tester_povm_names = get_tester_povm_names_method()
    tester_povms = []
    tester_povms_qutip = []
    for tester_povm_name in tester_povm_names:
        tester_povm = generate_povm_from_name(tester_povm_name, c_sys)
        tester_povms.append(tester_povm)
        tester_povms_qutip.append(convert_povm_quara_to_qutip(tester_povm))

    seed = 7896
    qst = StandardQst(
        tester_povms, on_para_eq_constraint=True, schedules="all", seed=seed
    )
    prob_dists_arrays = qst.calc_prob_dists(true_state)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    for estimator_name in ["linear", "least_squares"]:
        estimated_state_qutip = estimate_standard_qst_from_qutip(
            mode,
            num,
            tester_povms=tester_povms_qutip,
            empi_dists=prob_dists,
            estimator_name=estimator_name,
            schedules="all",
        )
        npt.assert_array_almost_equal(
            estimated_state_qutip.data.toarray(),
            true_state_qutip.data.toarray(),
            decimal=decimal,
        )


@pytest.mark.qutip
@pytest.mark.parametrize(
    ("mode", "num", "true_povm_name", "decimal"),
    [("qubit", 1, "z", 4), ("qutrit", 1, "z3", 4)],
)
def test_estimate_standard_povmt_from_qutip(mode, num, true_povm_name, decimal):
    c_sys = generate_composite_system(mode, num)
    true_povm = generate_povm_from_name(true_povm_name, c_sys)
    true_povm_qutip = convert_povm_quara_to_qutip(true_povm)

    get_tester_state_names_method_name = f"get_tester_state_names_{int(num)}{mode}"
    get_tester_state_names_method = eval(get_tester_state_names_method_name)
    tester_state_names = get_tester_state_names_method()
    tester_states = []
    tester_states_qutip = []
    for tester_state_name in tester_state_names:
        tester_state = generate_state_from_name(c_sys, tester_state_name)
        tester_states.append(tester_state)
        tester_states_qutip.append(convert_state_quara_to_qutip(tester_state))

    seed = 7896
    povmt = StandardPovmt(
        tester_states,
        true_povm.num_outcomes,
        on_para_eq_constraint=True,
        schedules="all",
        seed=seed,
    )
    prob_dists_arrays = povmt.calc_prob_dists(true_povm)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    for estimator_name in ["linear", "least_squares"]:
        estimated_povm_qutip = estimate_standard_povmt_from_qutip(
            mode,
            num,
            tester_states=tester_states_qutip,
            num_outcomes=true_povm.num_outcomes,
            empi_dists=prob_dists,
            estimator_name=estimator_name,
            schedules="all",
        )
        for estimated_item, true_item in zip(estimated_povm_qutip, true_povm_qutip):
            npt.assert_array_almost_equal(
                estimated_item.data.toarray(),
                true_item.data.toarray(),
                decimal=decimal,
            )


@pytest.mark.qutip
@pytest.mark.parametrize(
    ("mode", "num", "true_gate_name", "decimal"),
    [("qubit", 1, "identity", 4), ("qutrit", 1, "identity", 4)],
)
def test_estimate_standard_qpt_from_qutip(mode, num, true_gate_name, decimal):
    c_sys = generate_composite_system(mode, num)
    true_gate = generate_gate_from_gate_name(true_gate_name, c_sys)
    true_gate_qutip = convert_gate_quara_to_qutip(true_gate)

    get_tester_povm_names_method_name = f"get_tester_povm_names_{int(num)}{mode}"
    get_tester_povm_names_method = eval(get_tester_povm_names_method_name)
    tester_povm_names = get_tester_povm_names_method()
    tester_povms = []
    tester_povms_qutip = []
    for tester_povm_name in tester_povm_names:
        tester_povm = generate_povm_from_name(tester_povm_name, c_sys)
        tester_povms.append(tester_povm)
        tester_povms_qutip.append(convert_povm_quara_to_qutip(tester_povm))

    get_tester_state_names_method_name = f"get_tester_state_names_{int(num)}{mode}"
    get_tester_state_names_method = eval(get_tester_state_names_method_name)
    tester_state_names = get_tester_state_names_method()
    tester_states = []
    tester_states_qutip = []
    for tester_state_name in tester_state_names:
        tester_state = generate_state_from_name(c_sys, tester_state_name)
        tester_states.append(tester_state)
        tester_states_qutip.append(convert_state_quara_to_qutip(tester_state))

    seed = 7896
    qpt = StandardQpt(
        states=tester_states,
        povms=tester_povms,
        on_para_eq_constraint=True,
        schedules="all",
        seed=seed,
    )
    prob_dists_arrays = qpt.calc_prob_dists(true_gate)
    prob_dists = []
    for prob_dist in prob_dists_arrays:
        prob_dists.append((1, np.array(prob_dist)))

    for estimator_name in ["linear", "least_squares"]:
        estimated_gate_qutip = estimate_standard_qpt_from_qutip(
            mode,
            num,
            tester_states=tester_states_qutip,
            tester_povms=tester_povms_qutip,
            empi_dists=prob_dists,
            estimator_name=estimator_name,
            schedules="all",
        )
        npt.assert_array_almost_equal(
            estimated_gate_qutip.data.toarray(),
            true_gate_qutip.data.toarray(),
            decimal=decimal,
        )
