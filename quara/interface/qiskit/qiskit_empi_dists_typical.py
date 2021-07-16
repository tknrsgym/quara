from typing import List, Tuple
import numpy as np


def get_empi_dists_qiskit() -> np.ndarray:
    dists_qiskit = [0.864, 0.136, 0.844, 0.156, 0.49, 0.51]
    return dists_qiskit


def get_empi_dists_quara_list() -> List[Tuple[int, np.ndarray]]:
    dists_quara = [
        (1000, np.array([0.864, 0.136])),
        (2000, np.array([0.844, 0.156])),
        (3000, np.array([0.49, 0.51])),
    ]
    return dists_quara


def get_empi_dists_quara_int() -> List[Tuple[int, np.ndarray]]:
    dists_quara = [
        (1000, np.array([0.864, 0.136])),
        (1000, np.array([0.844, 0.156])),
        (1000, np.array([0.49, 0.51])),
    ]
    return dists_quara


def get_empi_dists_shots_list() -> List[int]:
    shots = [1000, 2000, 3000]
    return shots


def get_empi_dists_shots_int() -> int:
    shots = [1000, 1000, 1000]
    return shots


def get_empi_dists_label() -> List[int]:
    label = [2, 2, 2]
    return label
