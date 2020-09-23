from abc import abstractmethod
from typing import List

import numpy as np
import numpy.testing as npt
import pytest

from quara.data_analysis.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
)


def multiply_veca_vecb_matc(
    vec_a: np.array, vec_b: np.array, mat_c: np.array
) -> np.array:
    vec = np.dot(vec_a, vec_b @ mat_c)
    return vec


mat_p = np.array(
    [
        [1, 1, 0, 0],
        [1, -1, 0, 0],
        [1, 0, 1, 0],
        [1, 0, -1, 0],
        [1, 0, 0, 1],
        [1, 0, 0, -1],
    ],
    dtype=np.float64,
) / np.sqrt(2)


def _func_prob_dist(index: int):
    def _process(var: np.array) -> np.float64:
        return np.dot(mat_p[index], var)

    return _process


def func_prob_dists(x: int = None):
    # TODO 確率分布を返す関数はリスト長=3、確率を返す関数はリスト長=3なので、両立しない。要検討。
    funcs = []
    for index in range(len(mat_p)):
        funcs.append(_func_prob_dist(index))
    return funcs


def _func_gradient_prob_dist(index: int, x: int, alpha: int):
    def _process(var: np.array) -> np.float64:
        mat_p_index = index * 2 + x
        return mat_p[mat_p_index, alpha]

    return _process


def func_gradient_prob_dists():
    funcs = []
    for index in range(3):
        func_xs = []
        for x in range(2):
            func_alphas = []
            for alpha in range(4):
                func_alphas.append(_func_gradient_prob_dist(index, x, alpha))
            func_xs.append(func_alphas)
        funcs.append(func_xs)
    return funcs


def _func_hessian_prob_dist(index: int, x: int, alpha: int, beta: int):
    def _process(var: np.array):
        return 0.0

    return _process


def func_hessian_prob_dists():
    funcs = []
    for index in range(3):
        func_xs = []
        for x in range(2):
            func_alphas = []
            for alpha in range(4):
                func_betas = []
                for beta in range(4):
                    func_betas.append(_func_hessian_prob_dist(index, x, alpha, beta))
                func_alphas.append(func_betas)
            func_xs.append(func_alphas)
        funcs.append(func_xs)
    return funcs


prob_dists_q = [
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([0.5, 0.5], dtype=np.float64),
    np.array([1.0, 0.0], dtype=np.float64),
]


def func_weighted_probability_based_squared_error(
    q: List[np.array], p: List[np.array], W: List[np.array]
) -> np.array:
    tmp_errors = []
    for x_index in range(p):
        for xprime_index in range(p):
            value_x_index = p[x_index] - q[x_index]
            value_xprime_index = p[xprime_index] - q[xprime_index]
            error = np.dots(
                value_x_index, W[value_x_index, value_xprime_index] * value_xprime_index
            )
            tmp_errors.append(error)

    value = np.sum(tmp_errors)
    return value


def func_gradient_weighted_probability_based_squared_error(
    q: List[np.array], p: List[np.array], W: List[np.array], grad_p
) -> np.array:
    tmp_errors = []
    for x_index in range(p):
        for xprime_index in range(p):
            value_x_index = p[x_index] - q[x_index]
            value_xprime_index = p[xprime_index] - q[xprime_index]
            error = np.dots(
                value_x_index, W[value_x_index, value_xprime_index] * value_xprime_index
            )
            tmp_errors.append(error)

    value = 2 * np.sum(tmp_errors)
    return value


class TestProbabilityBasedLossFunction:
    def test_value(self):
        funcs = func_prob_dists(0)
        var = np.array([1, 0, 0, 1], dtype=np.float64) / np.sqrt(2)
        print("probs")
        for func in funcs:
            print(func(var))

        gradients = func_gradient_prob_dists()
        print("gradients")
        for i in range(3):
            for x in range(2):
                print(
                    i,
                    x,
                    gradients[i][x][0](var),
                    gradients[i][x][1](var),
                    gradients[i][x][2](var),
                    gradients[i][x][3](var),
                )

        hessians = func_hessian_prob_dists()
        print("hessians")
        for i in range(3):
            for x in range(2):
                for alpha in range(4):
                    print(
                        i,
                        x,
                        alpha,
                        hessians[i][x][alpha][0](var),
                        hessians[i][x][alpha][1](var),
                        hessians[i][x][alpha][2](var),
                        hessians[i][x][alpha][3](var),
                    )
