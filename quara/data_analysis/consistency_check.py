from typing import List
from quara.data_analysis import data_analysis, simulation
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)


def calc_mse_of_true_estimated(
    true_object: "QOperation",
    qtomography: "QTomography",
    estimator: "Estimator",
    loss: "ProbabilityBasedLossFunction" = None,
    loss_option: "ProbabilityBasedLossFunctionOption" = None,
    algo: "MinimizationAlgorithm" = None,
    algo_option: "MinimizationAlgorithmOption" = None,
) -> float:
    true_prob_dists = qtomography.generate_prob_dists_sequence(true_object)
    tmp_prob_dists = []
    for prob_dist in true_prob_dists:
        tmp_prob_dists.append((1, prob_dist))

    if isinstance(estimator, LossMinimizationEstimator):
        true_estimated = estimator.calc_estimate_sequence(
            qtomography,
            [tmp_prob_dists],
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
            is_computation_time_required=True,
        )
    else:
        true_estimated = estimator.calc_estimate_sequence(
            qtomography, [tmp_prob_dists], is_computation_time_required=True,
        )

    mse = data_analysis.calc_mse_qoperations(
        [true_estimated.estimated_qoperation], [true_object], with_std=False
    )

    return mse


def execute_consistency_check(
    simulation_setting: "StandardQTomographySimulation",
    estimation_results: List["EstimationResult"],
    eps=None,
    show_detail: bool = True,
) -> bool:
    eps = 10 ** (-16) if eps is None else eps
    result = calc_mse_of_true_estimated(
        true_object=simulation_setting.true_object,
        qtomography=estimation_results[0].qtomography,
        estimator=simulation_setting.estimator,
        loss=simulation_setting.loss,
        loss_option=simulation_setting.loss_option,
        algo=simulation_setting.algo,
        algo_option=simulation_setting.algo_option,
    )

    if show_detail:
        print(f"result={result}")
        print(f"eps={eps}")
        print(f"result < eps: {result < eps}")

    return result < eps
