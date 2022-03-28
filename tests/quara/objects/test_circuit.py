import numpy as np
import numpy.testing as npt
import pytest

from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.circuit import (
    Circuit,
)
from quara.objects.gate import convert_hs
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.mprocess import MProcess
from quara.objects.mprocess_typical import generate_mprocess_from_name
from quara.objects.state_typical import generate_state_from_name
from quara.utils.matrix_util import truncate_hs

def generate_qutrit_mprocess() -> MProcess:
    c_sys = generate_composite_system("qutrit", 1)
    kraus_matrices = [
        [
            np.array([[1,0,0],[0,0,0],[0,0,0]], dtype=np.complex128)
        ],
        [
            np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.complex128)
        ],
        [
            np.array([[0,0,0],[0,0,0],[0,0,1]], dtype=np.complex128)
        ]
    ]
    hss_cb = []
    size = c_sys.dim ** 2
    for kraus_matrices in kraus_matrices:
        tmp_hs = np.zeros((size, size), dtype=np.complex128)
        for kraus_matrix in kraus_matrices:
            tmp_hs += np.kron(kraus_matrix, kraus_matrix.conjugate())
        hss_cb.append(tmp_hs)
    hss = [
        truncate_hs(convert_hs(hs_cb, c_sys.comp_basis(), c_sys.basis()))
        for hs_cb in hss_cb
    ]
    return MProcess(
        hss=hss, c_sys=c_sys, is_physicality_required=True
    ) 

class TestCircuit:
    def test_add_gate_by_name_qubit(self):
        circuit = Circuit(4, "qubit")
        gate_names = ["x", "cx", "toffoli"]
        gate_ids = [[1], [0, 3], [1, 2, 3]]

        for name, ids in zip(gate_names, gate_ids):
            circuit.add_gate(ids=ids, gate_name=name)

        assert len(circuit) == 3
        for qobj, name, ids in zip(circuit, gate_names, gate_ids):
            assert qobj["Type"] == "Gate"
            assert qobj["TargetIds"] == ids
            assert qobj["Name"] == name

    @pytest.mark.skipci
    def test_add_gate_by_name_qutrit(self):
        circuit = Circuit(4, "qutrit")
        gate_names = ["12x180", "i12x90", "i01z90_i02x90"]
        gate_ids = [[1], [0, 3], [1,2]]
        
        for name, ids in zip(gate_names, gate_ids):
            circuit.add_gate(ids=ids, gate_name=name)

        assert len(circuit) == 3
        for qobj, name, ids in zip(circuit, gate_names, gate_ids):
            assert qobj["Type"] == "Gate"
            assert qobj["TargetIds"] == ids
            assert qobj["Name"] == name

    def test_add_gate(self):
        circuit = Circuit(4, "qubit")
        c_sys_1 = generate_composite_system("qubit", 1)
        gate = generate_gate_from_gate_name("x", c_sys_1)
        circuit.add_gate(ids=[0], gate=gate)

        assert len(circuit) == 1
        assert circuit[0]["Type"] == "Gate"
        assert circuit[0]["TargetIds"] == [0]

    def test_add_mprocess_by_name(self):
        circuit = Circuit(4, "qubit")
        mprocess_names = ["x-type1", "bell-type1", "z-type2"]
        mprocess_ids = [[1], [0, 3], [2]]

        for name, ids in zip(mprocess_names, mprocess_ids):
            circuit.add_mprocess(ids=ids, mprocess_name=name)

        assert len(circuit) == 3
        for qobj, name, ids in zip(circuit, mprocess_names, mprocess_ids):
            assert qobj["Type"] == "MProcess"
            assert qobj["TargetIds"] == ids
            assert qobj["Name"] == name

    def test_add_mprocess_qubit(self):
        circuit = Circuit(4, "qubit")
        c_sys_1 = generate_composite_system("qubit", 1)
        mprocess = generate_mprocess_from_name(c_sys_1, "x-type1")
        circuit.add_mprocess([0], mprocess=mprocess)

        assert len(circuit) == 1
        assert circuit[0]["Type"] == "MProcess"
        assert circuit[0]["TargetIds"] == [0]

    @pytest.mark.skipci
    def test_add_mprocess_qutrit(self):
        circuit = Circuit(2, "qutrit")
        mprocess = generate_qutrit_mprocess()
        circuit.add_mprocess([0], mprocess=mprocess)

        assert len(circuit) == 1
        assert circuit[0]["Type"] == "MProcess"
        assert circuit[0]["TargetIds"] == [0]

    def test_circuit_has_atleast_one_mprocess(self):
        circuit = Circuit(4, "qubit")
        with pytest.raises(ValueError):
            circuit.run(10, initial_state_mode="all_zero")

        circuit.add_mprocess([0], mprocess_name="z-type1")
        circuit.run(10, initial_state_mode="all_zero")

    @pytest.mark.skipci
    def test_initial_state_all_zero(self):
        circuit = Circuit(4, "qubit")
        for i in range(4):
            circuit.add_mprocess([i], mprocess_name="z-type1")
        res = circuit.run(100, initial_state_mode="all_zero")
        expected = np.array([1, 0], dtype=np.float64)
        for actual in res.empi_dists:
            npt.assert_equal(actual.ps, expected)

        circuit = Circuit(4, "qubit")
        for i in range(4):
            circuit.add_gate([i], gate_name="x")
            circuit.add_mprocess([i], mprocess_name="z-type2")
        res = circuit.run(100, initial_state_mode="all_zero")
        expected = np.array([0, 1], dtype=np.float64)
        for actual in res.empi_dists:
            npt.assert_equal(actual.ps, expected)

    @pytest.mark.skipci
    def test_initial_state_custom(self):
        circuit = Circuit(4, "qubit")
        state_names = ["z1", "x0", "x1", "y1"]
        c_sys = generate_composite_system("qubit", 1)
        states = [
            generate_state_from_name(c_sys, state_name) for state_name in state_names
        ]
        for i, state_name in enumerate(state_names):
            circuit.add_mprocess([i], mprocess_name=f"{state_name[0]}-type1")

        res = circuit.run(100, initial_states=states)
        expects = [[0, 1], [1, 0], [0, 1], [0, 1]]
        for actual, expected in zip(res.empi_dists, expects):
            expected_nd = np.array(expected, dtype=np.float64)
            npt.assert_equal(actual.ps, expected_nd)

    @pytest.mark.skipci
    def test_run_circuit_case1(self):
        circuit = Circuit(3, "qubit")
        # q_0 = |0>
        # NOP
        # q_1 = |+>
        circuit.add_gate([1], gate_name="hadamard")
        # q_2 = |0> + i|1>
        circuit.add_gate([2], gate_name="x90")
        circuit.add_gate([2], gate_name="z")

        circuit.add_mprocess([0], mprocess_name="z-type1")
        circuit.add_mprocess([1], mprocess_name="x-type1")
        circuit.add_mprocess([2], mprocess_name="y-type1")

        # check that circuit generates states of z0, x0, y0.
        res = circuit.run(100, initial_state_mode="all_zero")
        expects = [[1, 0], [1, 0], [1, 0]]
        for actual, expected in zip(res.empi_dists, expects):
            expected_nd = np.array(expected, dtype=np.float64)
            npt.assert_equal(actual.ps, expected_nd)

    @pytest.mark.skipci
    def test_run_circuit_case2(self):
        circuit = Circuit(3, "qubit")

        # check indices of multi bit gates
        circuit.add_gate([2], gate_name="x")
        circuit.add_gate([1, 0], gate_name="cx")
        circuit.add_gate([2, 1], gate_name="cx")

        circuit.add_mprocess([0], mprocess_name="z-type1")
        circuit.add_mprocess([1], mprocess_name="z-type1")
        circuit.add_mprocess([2], mprocess_name="z-type1")

        res = circuit.run(100, initial_state_mode="all_zero")
        expects = [[1, 0], [0, 1], [0, 1]]
        for actual, expected in zip(res.empi_dists, expects):
            expected_nd = np.array(expected, dtype=np.float64)
            npt.assert_equal(actual.ps, expected_nd)

    @pytest.mark.skipci
    def test_run_circuit_case3(self):
        circuit = Circuit(3, "qubit")

        # check for multi qubit mprocess
        circuit.add_gate([0], gate_name="hadamard")
        circuit.add_gate([0, 1], gate_name="cx")

        circuit.add_mprocess([0, 1], mprocess_name="bell-type1")

        res = circuit.run(100, initial_state_mode="all_zero")
        expects = [[1, 0, 0, 0]]
        for actual, expected in zip(res.empi_dists, expects):
            expected_nd = np.array(expected, dtype=np.float64)
            npt.assert_equal(actual.ps, expected_nd)

    @pytest.mark.skipci
    def test_run_circuit_case4(self):
        circuit = Circuit(3, "qutrit")

        # check for qutrit system
        circuit.add_gate([1], gate_name="01x180")
        circuit.add_gate([2], gate_name="02x180")
        mprocess = generate_qutrit_mprocess()
        circuit.add_mprocess([0], mprocess=mprocess)
        circuit.add_mprocess([1], mprocess=mprocess)
        circuit.add_mprocess([2], mprocess=mprocess)

        res = circuit.run(100, initial_state_mode="all_zero")
        expects = [[1,0,0],[0,1,0],[0,0,1]]
        for actual, expected in zip(res.empi_dists, expects):
            expected_nd = np.array(expected, dtype=np.float64)
            npt.assert_equal(actual.ps, expected_nd)

