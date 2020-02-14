from typing import List

import numpy as np

from quara.objects.matrix_basis import MatrixBasis
import quara.utils.matrix_util as mutil


def tensor_product(*elements):
    # 引数をリスト化
    element_list = []
    for element in elements:
        if type(element) == list:
            element_list.extend(element)
        else:
            element_list.append(element)

    # 要素が2以上でない場合はエラー
    assert len(element_list) >= 2

    # 再帰的にテンソル積を計算
    temp = element_list[-1]
    for elem in reversed(element_list[:-1]):
        temp = _tensor_product(elem, temp)
    return temp


def _tensor_product(elem1, elem2):
    # TODO 型ごとにテンソル積の計算を実装する
    if type(elem1) == MatrixBasis and type(elem1) == MatrixBasis:
        # TODO サンプル
        return elem1 + elem2
    else:
        raise ValueError(
            f"Unknown type combination! type=({type(elem1)}, {type(elem1)})"
        )
