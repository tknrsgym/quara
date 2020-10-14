from quara.data_analysis import data_analysis


def calc_mse_of_true_estimated(
    true_object: "QOperation", qtomography: "QTomography", estimator: "Estimator"
) -> float:
    true_prob_dists = qtomography.generate_prob_dists_sequence(true_object)
    tmp_prob_dists = []
    for prob_dist in true_prob_dists:
        tmp_prob_dists.append((1, prob_dist))

    true_estimated = estimator.calc_estimate_sequence(
        qtomography=qtomography,
        empi_dists_sequence=[tmp_prob_dists],
        is_computation_time_required=True,
    )

    mses, *_ = data_analysis.convert_to_series([true_estimated], true_object)

    return mses[0]
