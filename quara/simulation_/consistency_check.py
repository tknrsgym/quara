from typing import List, Tuple
from quara.data_analysis import data_analysis
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
) -> Tuple[float, List["EstimationResult"]]:
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
            qtomography,
            [tmp_prob_dists],
            is_computation_time_required=True,
        )

    mse = data_analysis.calc_mse_qoperations(
        [true_estimated.estimated_qoperation], [true_object], with_std=False
    )

    return (mse, true_estimated)


def execute_consistency_check(
    simulation_setting: "StandardQTomographySimulationSetting",
    estimation_results: List["EstimationResult"],
    eps=None,
    show_detail: bool = True,
) -> dict:
    if eps is None:
        eps = 10 ** (-10)
    value, estimation_result = calc_mse_of_true_estimated(
        true_object=simulation_setting.true_object,
        qtomography=estimation_results[0].qtomography,
        estimator=simulation_setting.estimator,
        loss=simulation_setting.loss,
        loss_option=simulation_setting.loss_option,
        algo=simulation_setting.algo,
        algo_option=simulation_setting.algo_option,
    )

    # numpy.bool_ -> bool to serialize to json
    result = bool(value < eps)
    if show_detail:
        print(f"[{'OK' if result else 'NG'}] Consistency Check")
        print(f"value={value}")
        print(f"eps={eps}")
        print(f"result(value < eps): {result}")

    param = estimation_results[0].estimated_qoperation.on_para_eq_constraint
    possibly_ok = result
    to_be_checked = not result if param else False

    data = dict()
    data["possibly_ok"] = possibly_ok
    data["to_be_checked"] = to_be_checked
    data["squared_error_to_true"] = value
    data["estimation_result"] = estimation_result
    return data
