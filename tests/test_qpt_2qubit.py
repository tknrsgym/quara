import os

import matlab.engine
import numpy as np


def load_state_list(path: str, dim: int, num_state: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    state_list = np.reshape(raw_data, (num_state, dim * dim))
    print(len(state_list))
    return state_list


def load_povm_list(
    path: str, dim: int, num_povm: int, num_outcome: int
) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.complex128)
    povm_list = np.reshape(raw_data, (num_povm, num_outcome, dim * dim))
    return povm_list


def load_schedule(path: str) -> (int, np.ndarray):
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.uint16)
    num_schedule, _ = raw_data.shape
    return num_schedule, raw_data


def load_empi_list(path: str, num_schedule: int = None) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    _, num_outcome = raw_data.shape
    return num_outcome, raw_data


def load_weight_list(path: str, n_schedule: int, num_outcome: int) -> np.ndarray:
    raw_data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    weight_list = np.reshape(raw_data, (n_schedule, num_outcome, num_outcome))
    return weight_list


def main(settings: dict) -> np.ndarray:
    print("--- load schedule ---")
    num_schedule, schedule_np = load_schedule(settings["path_schedule"])
    print(f"num_schedule={num_schedule}")
    schedule_ml = matlab.uint64(schedule_np.tolist())
    print(schedule_ml)

    print("--- load empi list ---")
    num_outcome, empi_list_np = load_empi_list(settings["path_empi"], num_schedule)
    empi_list_ml = matlab.double(empi_list_np.tolist())
    print(f"num_outcome={num_outcome}")
    print(empi_list_ml)

    print("--- load state list ---")
    state_list_np = load_state_list(
        settings["path_state"], settings["dim"], settings["num_state"]
    )
    state_list_ml = matlab.double(state_list_np.tolist(), is_complex=True)
    print(state_list_ml)

    print("--- load povm list ---")
    povm_list_np = load_povm_list(
        settings["path_povm"],
        settings["dim"],
        settings["num_povm"],
        settings["num_outcome"],
    )
    povm_list_ml = matlab.double(povm_list_np.tolist(), is_complex=True)
    print(povm_list_ml)

    print("--- load weight list ---")
    weight_list_np = load_weight_list(
        settings["path_weight"], num_schedule, settings["num_outcome"]
    )
    weight_list_ml = matlab.double(weight_list_np.tolist())
    print(weight_list_ml)

    """
    eng = matlab.engine.start_matlab()
    choi_matrix = eng.xxx(
        state_list_ml, povm_list_ml, schedule_ml, empi_list_ml, weight_list_ml
    )
    eng.quit()
    print(choi_matrix)
    """


if __name__ == "__main__":
    csv_path = os.path.dirname(__file__) + "/data/"
    settings = {
        "dim": 2 ** 2,
        "num_state": 16,
        "num_povm": 9,
        "num_outcome": 4,
        "path_state": csv_path + "tester_2qubit_state.csv",
        "path_povm": csv_path + "tester_2qubit_povm.csv",
        "path_schedule": csv_path + "schedule_2qubit.csv",
        "path_empi": csv_path + "listEmpiDist_4valued.csv",
        "path_weight": csv_path + "weight_4valued_uniform.csv",
    }
    main(settings)
