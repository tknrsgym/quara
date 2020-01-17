import logging
import os

from quara.protocol import simple_qpt
from quara.utils import matrix_util


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
choi, obj_value = simple_qpt.execute_from_csv(settings)
print("--- result ---")
print(f"choi={choi}")
print(f"obj_value={obj_value}")

print("--- validation ---")
print(f"is_hermitian={matrix_util.is_hermitian(choi)}")
print(f"is_positive_semidefinite={matrix_util.is_positive_semidefinite(choi)}")
print(f"is_tp={matrix_util.is_tp(choi, settings['dim'])}")
