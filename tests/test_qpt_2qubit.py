import logging
import os

from quara.protocol import simple_qpt
from quara.utils import matrix_util


# logging setting
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# setting for simple QPT
csv_path = os.path.dirname(__file__) + "/data/"
settings = {
    "dim": 2 ** 2,
    "num_state": 16,
    "num_povm": 9,
    "num_outcome": 4,
    "path_state": csv_path + "tester_2qubit_state.csv",
    "path_povm": csv_path + "tester_2qubit_povm.csv",
    "path_schedule": csv_path + "schedule_2qubit_start_from_0.csv",
    "path_empi": csv_path + "listEmpiDist_4valued.csv",
    "path_weight": csv_path + "weight_4valued_uniform.csv",
}

# execute simple QPT
choi, wsd = simple_qpt.execute_from_csv(settings)

# print result
print("--- result ---")
print(f"Choi matrix:\n{choi}")
print(f"weighted squared distance={wsd}")

# validate result
print("\n--- validation ---")
print(f"is Hermitian? : {matrix_util.is_hermitian(choi)}")
print(f"is positive semidefinite? : {matrix_util.is_positive_semidefinite(choi)}")
print(f"is TP? : {matrix_util.is_tp(choi, settings['dim'])}")
