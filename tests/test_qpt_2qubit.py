import os

from quara.protocol import simple_qpt


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
simple_qpt.execute(settings)
