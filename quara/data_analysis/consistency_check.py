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
