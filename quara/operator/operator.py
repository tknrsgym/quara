import itertools
from typing import List

import numpy as np

import quara.objects.matrix_basis as m_basis
from quara.objects.matrix_basis import MatrixBasis
from quara.objects.povm import Povm
import quara.utils.matrix_util as mutil


def tensor_product(*elements):
    # 引数をリスト化
    element_list = _to_list(*elements)

    # 再帰的にテンソル積を計算(リストの前から後ろに計算する)
    temp = element_list[0]
    for elem in element_list[1:]:
        temp = _tensor_product(temp, elem)
    return temp


def _tensor_product(elem1, elem2):
    # TODO 型ごとにテンソル積の計算を実装する
    if type(elem1) == MatrixBasis and type(elem2) == MatrixBasis:
        mat_list = [
            np.kron(val1, val2) for val1, val2 in itertools.product(elem1, elem2)
        ]
        basis = MatrixBasis(mat_list)
        return basis
    elif type(elem1) == Povm and type(elem2) == Povm:
        return _tensor_product_povm(elem1, elem2)
    else:
        raise ValueError(
            f"Unknown type combination! type=({type(elem1)}, {type(elem2)})"
        )


def _tensor_product_povm(elem1: Povm, elem2: Povm):
    pass


def composite(*elements):
    # 引数をリスト化
    element_list = _to_list(*elements)

    # 再帰的に合成を計算(リストを後ろから前に計算する)
    temp = element_list[-1]
    for elem in reversed(element_list[:-1]):
        temp = _composite(elem, temp)
    return temp


def _composite(elem1, elem2):
    # TODO 要実装
    pass


def _to_list(*elements):
    # 引数をリスト化
    element_list = []
    for element in elements:
        if type(element) == list:
            element_list.extend(element)
        else:
            element_list.append(element)

    # 要素が2以上でない場合はエラー
    assert len(element_list) >= 2

    return element_list

