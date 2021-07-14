from typing import Union
import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.mprocess import MProcess


def get_mprocess_names():
    return ["z"]  # TODO: implement


def generate_mprocess_hss_from_name(mprocess_name: str):
    if mprocess_name not in get_mprocess_names():
        message = f"mprocess_name is out of range."
        raise ValueError(message)

    typical_names = get_mprocess_names()
    if mprocess_name in typical_names:
        method_name = f"get_mprocess_{mprocess_name}_hss"
        method = eval(method_name)
        return method()


def get_mprocess_z_hss():
    hs_0 = (1 / 2) * np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])
    hs_1 = (1 / 2) * np.array(
        [[1, 0, 0, -1], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 0, 0, 1]]
    )
    hss = [hs_0, hs_1]
    return hss


def generate_mprocess_from_name(c_sys: CompositeSystem, mprocess_name: str) -> MProcess:
    # TODO:
    # hss = generate_mprocess_hss_from_name(c_sys.basis(), mprocess_name)
    hss = generate_mprocess_hss_from_name(mprocess_name)
    mprocess = MProcess(hss=hss, c_sys=c_sys)
    return mprocess


def generate_mprocess_object_from_mprocess_name_object_name(
    mprocess_name: str, object_name: str, c_sys: CompositeSystem = None
) -> Union[MProcess, np.ndarray]:
    expected_object_names = [
        "mprocess",
    ]

    if object_name not in expected_object_names:
        raise ValueError("object_name is out of range.")
    if object_name == "mprocess":
        return generate_mprocess_from_name(c_sys, mprocess_name)
