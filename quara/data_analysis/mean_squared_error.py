from typing import List
from quara.data_analysis import data_analysis
from quara.utils import matrix_util


def check_mse_of_empirical_distributions(
    simulation_setting, estimation_results, show_detail: bool = True
) -> bool:
    qtomography = estimation_results[0].qtomography
    num_data = estimation_results[0].num_data
    num_schedules = qtomography.num_schedules
    n_rep = len(estimation_results)

    # MSE of Empirical Distributions
    empi_dists = data_analysis.extract_empi_dists(estimation_results)
    xs_list_list = empi_dists
    ys_list_list = [
        [qtomography.calc_prob_dists(simulation_setting.true_object)] * n_rep
    ] * len(num_data)

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
            simulation_setting.true_object, [num] * num_schedules
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

def check_mse_of_estimators(simulation_setting: "StandardQTomographySimulation",
    estimation_results: List["EstimationResult"], show_detail: bool=True) -> bool:
    
    pass