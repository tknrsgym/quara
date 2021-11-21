from typing import List
from quara.data_analysis import data_analysis
from quara.utils import matrix_util

from quara.loss_function.weighted_relative_entropy import WeightedRelativeEntropy
from quara.protocol.qtomography.standard.linear_estimator import LinearEstimator
from quara.protocol.qtomography.standard.projected_linear_estimator import (
    ProjectedLinearEstimator,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)
from quara.simulation.standard_qtomography_simulation import (
    SimulationResult,
    generate_empi_dists_and_calc_estimate,
)


def check_mse_of_empirical_distributions(
    simulation_result: SimulationResult, show_detail: bool = True
) -> bool:
    num_data = simulation_result.simulation_setting.num_data
    n_rep = simulation_result.simulation_setting.n_rep
    qtomography = simulation_result.qtomography
    num_schedules = qtomography.num_schedules

    parameter = qtomography.on_para_eq_constraint
    true_object_copied = data_analysis._recreate_qoperation(
        simulation_result.simulation_setting.true_object,
        on_para_eq_constraint=parameter,
    )

    # MSE of Empirical Distributions
    empi_dists_sequences = data_analysis.extract_empi_dists_sequences(
        simulation_result.empi_dists_sequences
    )
    xs_list_list = empi_dists_sequences
    ys_list_list = [[qtomography.calc_prob_dists(true_object_copied)] * n_rep] * len(
        num_data
    )

    empi_dist_mses = []
    sigma_list = []
    for i in range(len(num_data)):
        mse, std = matrix_util.calc_mse_prob_dists(xs_list_list[i], ys_list_list[i])
        empi_dist_mses.append(mse)
        sigma_list.append(std)

    # MSE of Analytical
    analytical_mses = []
    for num in num_data:
        analytical_mse = qtomography.calc_mse_empi_dists_analytical(
            true_object_copied, [num] * num_schedules
        )
        analytical_mses.append(analytical_mse)

    result_list = []
    for i in range(len(num_data)):
        empi_dist_mse = empi_dist_mses[i]
        analytical_mse = analytical_mses[i]
        eps = sigma_list[i] * 3
        result = abs(empi_dist_mse - analytical_mse) < eps
        result_list.append(result)
        if show_detail:
            result_text = "OK" if result else "NG"
            text = f"[{result_text}] N={num_data[i]}"
            text += f"\n|MSE_EmpiDist - MSE_Analytical| = {abs(empi_dist_mse - analytical_mse)}"
            text += f"\neps = {eps}"
            print(text)

    if False in result_list:
        return False
    else:
        return True


def check_mse_of_estimators(
    simulation_setting: "StandardQTomographySimulation",
    estimation_results: List["EstimationResult"],
    qtomography,
    show_detail: bool = True,
) -> bool:
    def _is_maximum_likelihood(simulation_setting) -> bool:
        if (type(simulation_setting.estimator) == LossMinimizationEstimator) and (
            type(simulation_setting.loss) == WeightedRelativeEntropy
        ):
            return True
        return False

    if type(simulation_setting.estimator) == LinearEstimator:
        result = compare_to_analytical(
            simulation_setting, estimation_results, qtomography, show_detail
        )
    elif type(
        simulation_setting.estimator
    ) == ProjectedLinearEstimator or _is_maximum_likelihood(simulation_setting):
        result = compare_to_linear(
            simulation_setting, estimation_results, qtomography, show_detail
        )
    else:
        message = f"Estimator must be LinearEstimator, ProjectedLinearEstimator, or Maximum-likelihood."
        raise TypeError(message)

    return result


def compare_to_analytical(
    simulation_setting,
    estimation_results: List["EstimationResult"],
    qtomography,
    show_detail: bool = True,
) -> bool:
    if type(simulation_setting.estimator) != LinearEstimator:
        raise TypeError(
            f"simulation_setting.estimator must be LinearEstimator, not {type(simulation_setting.estimator)}"
        )

    num_data = simulation_setting.num_data
    parameter = qtomography.on_para_eq_constraint
    true_object_copied = data_analysis._recreate_qoperation(
        simulation_setting.true_object, on_para_eq_constraint=parameter
    )

    # MSE_Linear
    mses, sds, _ = data_analysis.convert_to_series(
        estimation_results, true_object_copied
    )

    # MSE_Analytical
    analytical_mses = []
    for num in num_data:
        analytical_mse = qtomography.calc_mse_linear_analytical(
            true_object_copied, [num] * qtomography.num_schedules
        )
        analytical_mses.append(analytical_mse)

    results = []
    for i in range(len(num_data)):
        mse_linear = mses[i]
        sigma = sds[i]
        mse_analytical = analytical_mses[i]
        eps = 3 * sigma

        result = abs(mse_linear - mse_analytical) < eps
        results.append(result)

        if show_detail:
            result_text = "OK" if result else "NG"
            text = f"[{result_text}] N={num_data[i]}"
            text += (
                f"\n|MSE_Linear - MSE_Analytical| = {abs(mse_linear - mse_analytical)}"
            )
            text += f"\neps = {eps}"
            print(text)

    if False in results:
        return False
    else:
        return True


def compare_to_linear(
    simulation_setting, estimation_results, qtomography, show_detail: bool = True
) -> bool:
    # calc mse
    mses, *_ = data_analysis.convert_to_series(
        estimation_results, simulation_setting.true_object
    )

    # calc mse of LinearEstimator
    num_data = simulation_setting.num_data
    n_rep = simulation_setting.n_rep
    parameter = qtomography.on_para_eq_constraint
    true_object_copied = data_analysis._recreate_qoperation(
        simulation_setting.true_object, on_para_eq_constraint=parameter
    )

    # generate empi dists and calc estimate
    linear_estimator = LinearEstimator()
    sim_result = generate_empi_dists_and_calc_estimate(
        qtomography=qtomography,
        true_object=true_object_copied,
        num_data=num_data,
        estimator=linear_estimator,
        iteration=n_rep,
    )
    # calc
    linear_mses, *_ = data_analysis.convert_to_series(
        sim_result.estimation_results, true_object_copied
    )

    # compare
    results = []
    for i in range(len(num_data)):
        mse = mses[i]
        linear_mse = linear_mses[i]
        result = mse <= linear_mse
        results.append(result)

        if show_detail:
            result_text = "OK" if result else "NG"
            text = f"[{result_text}] N={num_data[i]}"
            estimator_name = simulation_setting.estimator.__class__.__name__
            text += f"\nMSE_Linear - MSE_{estimator_name} = {linear_mse - mse}"
            print(text)

    if False in results:
        return False
    else:
        return True
