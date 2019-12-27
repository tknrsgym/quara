import os

import matlab
import numpy as np
import pytest


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
    dim = 2 ** 1  # 2**qubits
    num_state = 4
    num_povm = 3
    csv_path = os.path.dirname(__file__) + "/data/"

    print("--- load state list ---")
    state_np = load_state_list(csv_path + "tester_1qubit_state.csv", dim, num_state)
    state_ml = matlab.double(state_np.tolist(), is_complex=True)
    print(state_ml)

    print("--- load povm list ---")
    m, povm_np = load_povm_list(csv_path + "tester_1qubit_povm.csv", dim, num_povm)
    povm_ml = matlab.double(povm_np.tolist(), is_complex=True)
    print(f"M={m}")
    print(povm_ml)

    print("--- load schedule ---")
    num_schedule, schedule_np = load_schedule(csv_path + "schedule_1qubit.csv")
    print(f"n_schedule={num_schedule}")
    schedule_ml = matlab.uint64(schedule_np.tolist())
    print(schedule_ml)

    print("--- load emp list ---")
    emp_list_np = load_emp_list(
        csv_path + "listEmpiDist_2valued.csv", dim, num_schedule
    )
    emp_list_ml = matlab.double(emp_list_np.tolist())
    print(emp_list_ml)

    print("--- load weight list ---")
    weight_list_np = load_weight_list(
        csv_path + "weight_2valued_uniform.csv", num_schedule, m
    )
    weight_list_ml = matlab.double(weight_list_np.tolist())
    print(weight_list_ml)
