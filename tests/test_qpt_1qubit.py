import os
from pathlib import Path

import matlab
import matlab.engine
import numpy as np
import pytest

# from quara.engine.matlabengine import MatlabEngine


def load_state_list(path: str, dim: int, num_state: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)

    # TODO: check row of data
    if raw_data.shape[0] != num_state * dim:
        raise ValueError(
            "Invalid number of data in state list"
        )  # TODO: consider error message

    state_list = np.reshape(raw_data, (num_state, dim * dim))
    return state_list


def load_povm_list(path: str, dim: int, num_povm: int) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    m = int(len(raw_data) / (dim * num_povm))

    povm_list = np.reshape(raw_data, (num_povm, m, dim * dim))
    return m, povm_list


def load_schedule(path: str) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_schedule, _ = raw_data.shape
    return num_schedule, raw_data


def load_emp_list(path: str, dim: int, num_schedule: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    emp_list = np.reshape(raw_data, (num_schedule, dim))
    return emp_list


def load_weight_list(path: str, num_schedule: int, m: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    weight_list = np.reshape(raw_data, (num_schedule, m, m))
    return weight_list


def test_load_state_list_invalid_num():
    path = os.path.dirname(__file__) + "/data/tester_1qubit_state.csv"
    dim = 2 ** 1
    state_np = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    invalid_num_state = state_np.shape[0] // dim + 1  # make invalid data

    with pytest.raises(ValueError):
        _ = load_state_list(path, dim, invalid_num_state)


if __name__ == "__main__":

    eng = matlab.engine.start_matlab()

    dim = 2 ** 1  # 2**qubits
    num_state = 4
    num_povm = 3
    this_pypath = os.path.abspath(__file__)
    matlab_func_path = os.path.dirname(this_pypath)
    csv_path = Path(os.path.dirname(this_pypath)) / "data"

    print("--- load state list ---")
    state_np = load_state_list(csv_path / "tester_1qubit_state.csv", dim, num_state)
    state_ml = matlab.double(state_np.tolist(), is_complex=True)
    print(state_ml)

    # eng.List_state_from_python(state_ml)

    print("--- load povm list ---")
    m, povm_np = load_povm_list(csv_path / "tester_1qubit_povm.csv", dim, num_povm)
    povm_ml = matlab.double(povm_np.tolist(), is_complex=True)
    print(f"M={m}")
    print(povm_ml)

    # eng.List_povm_from_python(povm_ml)

    print("--- load schedule ---")
    num_schedule, schedule_np = load_schedule(csv_path / "schedule_1qubit.csv")
    print(f"n_schedule={num_schedule}")
    schedule_ml = matlab.uint64(schedule_np.tolist())
    print(schedule_ml)

    # eng.List_schedule_from_python(schedule_ml)

    print("--- load emp list ---")
    emp_list_np = load_emp_list(
        csv_path / "listEmpiDist_2valued.csv", dim, num_schedule
    )
    emp_list_ml = matlab.double(emp_list_np.tolist())
    print(emp_list_ml)

    # eng.List_empiDist_from_python(emp_list_ml)

    print("--- load weight list ---")
    weight_list_np = load_weight_list(
        csv_path / "weight_2valued_uniform.csv", num_schedule, m
    )
    weight_list_ml = matlab.double(weight_list_np.tolist())
    print(weight_list_ml)

    # eng.List_weight_from_python(weight_list_ml)

    eps_sedumi = 0.0  # matlab.double(0.0)
    int_verbose = 1  # matlab.uint8(1)
    [Choi, value] = eng.simple_qpt(
        dim,
        state_ml,
        povm_ml,
        schedule_ml,
        weight_list_ml,
        emp_list_ml,
        eps_sedumi,
        int_verbose,
    )
    print(Choi)
    print(value)

    eng.quit()

    """with MatlabEngine() as engine:
        engine.check_pass_from_python_to_matlab(
            state_ml, nargout=0,
        ) """

    print("completed")
