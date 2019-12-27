import os

import matlab
import numpy as np


def load_state_list(path: str, dim: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    num_of_data = int(len(raw_data) / dim)
    # TODO: check row of data
    state_list = np.reshape(raw_data, (num_of_data, dim * dim))
    return state_list


def load_povm_list(path: str, dim: int, Np: int) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    M = int(len(raw_data) / (dim * Np))

    povm_list = np.reshape(raw_data, (Np, M, dim * dim))
    return M, povm_list


def load_schedule(path: str) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    Nsche, _ = raw_data.shape
    return Nsche, raw_data


def load_emp_list(path: str, dim: int, Nsche: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    emp_list = np.reshape(raw_data, (Nsche, dim))
    return emp_list


def load_weight_list(path: str, Nsche: int, M: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)

    weight_list = np.reshape(raw_data, (Nsche, M, M))
    return weight_list


if __name__ == "__main__":
    dim = 2 ** 1  # 2**qubits
    Np = 3
    csv_path = os.path.dirname(__file__) + "/data/"

    print("--- load state list ---")
    state_np = load_state_list(csv_path + "tester_1qubit_state.csv", dim)
    state_ml = matlab.double(state_np.tolist(), is_complex=True)
    print(state_ml)

    print("--- load povm list ---")
    M, povm_np = load_povm_list(csv_path + "tester_1qubit_povm.csv", dim, Np)
    povm_ml = matlab.double(povm_np.tolist(), is_complex=True)
    print(f"M={M}")
    print(povm_ml)

    print("--- load schedule ---")
    Nsche, schedule_np = load_schedule(csv_path + "schedule_1qubit.csv")
    print(f"Nsche={Nsche}")
    schedule_ml = matlab.uint64(schedule_np.tolist())
    print(schedule_ml)

    print("--- load emp list ---")
    emp_list_np = load_emp_list(csv_path + "listEmpiDist_2valued.csv", dim, Nsche)
    emp_list_ml = matlab.double(emp_list_np.tolist())
    print(emp_list_ml)

    print("--- load weight list ---")
    weight_list_np = load_weight_list(csv_path + "weight_2valued_uniform.csv", Nsche, M)
    weight_list_ml = matlab.double(weight_list_np.tolist())
    print(weight_list_ml)
