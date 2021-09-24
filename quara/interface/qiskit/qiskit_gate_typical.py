from quara.objects.composite_system import CompositeSystem
from typing import List, Union
import numpy as np
from qiskit.circuit.library.standard_gates import rzx, x, y, z, swap
from qiskit.quantum_info.operators.channel import Choi
from quara.objects.composite_system_typical import generate_composite_system
from quara.objects.gate_typical import generate_gate_from_gate_name
from quara.objects.operators import compose_qoperations, tensor_product
from quara.interface.qiskit.conversion import (
    calc_swap_matrix,
    convert_empi_dists_qiskit_to_quara,
)
from quara.objects.gate import Gate


def get_qiskit_gate_names_1qubit() -> List[str]:
    return ["x", "y", "z"]


def get_qiskit_gate_names_2qubit() -> List[str]:
    return ["cx"]


def get_qiskit_gate_names_3qubit() -> List[str]:
    return ["toffoli"]


def generate_qiskit_gate_from_gate_name(
    gate_name: str, ids: Union[None, List[int]]
) -> np.ndarray:
    if gate_name == "x":
        gate = x.XGate()
        mat = Choi(gate)

    elif gate_name == "y":
        gate = y.YGate()
        mat = Choi(gate)

    elif gate_name == "z":
        gate = z.ZGate()
        mat = Choi(gate)

    elif gate_name == "cx":
        if ids[0] < ids[1]:
            gate = x.CXGate(ctrl_state=0)
            mat = Choi(gate)
        elif ids[1] < ids[0]:
            gate = x.CXGate(ctrl_state=1)
            mat = Choi(gate)

    elif gate_name == "toffoli":
        if ids == [2, 0, 1] or ids == [2, 1, 0]:
            gate = x.CCXGate(ctrl_state=3)
            mat = Choi(gate)

        elif ids == [1, 2, 0] or ids == [0, 2, 1]:
            gate = x.CCXGate(ctrl_state=0)
            mat = Choi(gate)

        elif ids == [0, 1, 2] or ids == [1, 0, 2]:
            gate = x.CCXGate(ctrl_state=1)
            mat = Choi(gate)

    return mat


def generate_quara_gate_from_ids(
    gate_name: str,
    c_sys: CompositeSystem,
    ids: List[int],
) -> Gate:

    if gate_name == "cx":
        c_sys_0 = CompositeSystem([c_sys[0]])
        c_sys_1 = CompositeSystem([c_sys[1]])
        x0 = generate_gate_from_gate_name("x", c_sys_0)
        x1 = generate_gate_from_gate_name("x", c_sys_1)
        xx = tensor_product(x0, x1)

        if ids == [0, 1]:
            qua = generate_gate_from_gate_name("cx", c_sys, ids=[1, 0])
            gate = compose_qoperations(xx, compose_qoperations(qua, xx))
        else:
            qua = generate_gate_from_gate_name(gate_name, c_sys, ids)
            gate = qua

    elif gate_name == "toffoli":
        c_sys_0 = CompositeSystem([c_sys[0]])
        c_sys_1 = CompositeSystem([c_sys[1]])
        c_sys_2 = CompositeSystem([c_sys[2]])
        x0 = generate_gate_from_gate_name("x", c_sys_0)
        x1 = generate_gate_from_gate_name("x", c_sys_1)
        x2 = generate_gate_from_gate_name("x", c_sys_2)

        if ids == [0, 1, 2] or ids == [1, 0, 2]:
            qua = generate_gate_from_gate_name("toffoli", c_sys, ids=[0, 1, 2])
            i0 = generate_gate_from_gate_name("identity", c_sys_0)
            xx = tensor_product(i0, tensor_product(x1, x2))
            gate = compose_qoperations(xx, compose_qoperations(qua, xx))
        elif ids == [0, 2, 1] or ids == [1, 2, 0]:
            qua = generate_gate_from_gate_name("toffoli", c_sys, ids=[0, 2, 1])
            i0 = generate_gate_from_gate_name("identity", c_sys_0)
            xx = tensor_product(i0, tensor_product(x1, x2))
            gate = compose_qoperations(xx, compose_qoperations(qua, xx))
        elif ids == [2, 0, 1] or ids == [2, 1, 0]:
            qua = generate_gate_from_gate_name("toffoli", ids=[2, 0, 1])
            gate = qua

    return gate


def get_swap_matrix_2dim() -> np.ndarray:
    mat = np.zeros((4, 4), dtype=np.complex64)
    mat[0, 0] = 1
    mat[1, 2] = 1
    mat[2, 1] = 1
    mat[3, 3] = 1
    return mat


def get_swap_matrix_3dim() -> np.ndarray:
    mat = np.zeros((9, 9), dtype=np.complex64)
    mat[0, 0] = 1
    mat[1, 3] = 1
    mat[2, 6] = 1
    mat[3, 1] = 1
    mat[4, 4] = 1
    mat[5, 7] = 1
    mat[6, 2] = 1
    mat[7, 5] = 1
    mat[8, 8] = 1
    return mat
