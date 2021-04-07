from typing import List, Optional, Union
import copy
from collections import Counter

from tqdm import tqdm

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qtomography_estimator import (
    StandardQTomographyEstimator,
    StandardQTomographyEstimationResult,
)
from quara.minimization_algorithm.minimization_algorithm import (
    MinimizationAlgorithm,
    MinimizationAlgorithmOption,
)
from quara.loss_function.probability_based_loss_function import (
    ProbabilityBasedLossFunction,
    ProbabilityBasedLossFunctionOption,
)
from quara.protocol.qtomography.standard.loss_minimization_estimator import (
    LossMinimizationEstimator,
)


class StandardQTomographySimulationSetting:
    def __init__(
        self,
        name: str,
        true_object: "QOperation",
        tester_objects: List["QOperation"],
        estimator: "Estimator",
        seed: int,
        n_rep: int,
        num_data: List[int],
        schedules: Union[str, List[List[int]]],
        loss=None,
        loss_option=None,
        algo=None,
        algo_option=None,
    ) -> None:
        self.name = name
        self.true_object = copy.copy(true_object)
        self.tester_objects = copy.copy(tester_objects)
        self.estimator = copy.copy(estimator)
        self.loss = copy.copy(loss)
        self.loss_option = loss_option
        self.algo = copy.copy(algo)
        self.algo_option = algo_option

        self.seed = seed
        self.n_rep = n_rep
        self.num_data = num_data

        self.schedules = schedules

    def __str__(self):
        desc = f"Name: {self.name}"
        desc += f"\nTrue Object: {self.true_object}"

        counter = Counter([t.__class__.__name__ for t in self.tester_objects])
        desc += "\nTester Objects" + ", ".join(
            [f"{k} x {v}" for k, v in counter.items()]
        )

        desc += f"\nn_rep: {self.n_rep}"
        desc += f"\nnum_data: {self.num_data}"
        desc += f"\nEstimator: {self.estimator.__class__.__name__}"
        loss = None if self.loss is None else self.loss.__class__.__name__
        desc += f"\nLoss: {loss}"
        algo = None if self.algo is None else self.algo.__class__.__name__
        desc += f"\nAlgo: {algo}"
        return desc


# common
def execute_simulation(
    qtomography: "StandardQTomography",
    simulation_setting: StandardQTomographySimulationSetting,
) -> List[StandardQTomographyEstimationResult]:
    estimation_results = generate_empi_dists_and_calc_estimate(
        qtomography=qtomography,
        true_object=simulation_setting.true_object,
        num_data=simulation_setting.num_data,
        estimator=simulation_setting.estimator,
        loss=simulation_setting.loss,
        loss_option=simulation_setting.loss_option,
        algo=simulation_setting.algo,
        algo_option=simulation_setting.algo_option,
        iteration=simulation_setting.n_rep,
    )
    return estimation_results


# common
def _generate_empi_dists_and_calc_estimate(
    qtomography: "StandardQTomography",
    true_object: QOperation,
    num_data: List[int],
    estimator=StandardQTomographyEstimator,
    loss: ProbabilityBasedLossFunction = None,
    loss_option: ProbabilityBasedLossFunctionOption = None,
    algo: MinimizationAlgorithm = None,
    algo_option: MinimizationAlgorithmOption = None,
) -> StandardQTomographyEstimationResult:
    empi_dists_seq = qtomography.generate_empi_dists_sequence(true_object, num_data)

    if isinstance(estimator, LossMinimizationEstimator):
        result = estimator.calc_estimate_sequence(
            qtomography,
            empi_dists_seq,
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
            is_computation_time_required=True,
        )
    else:
        result = estimator.calc_estimate_sequence(
            qtomography, empi_dists_seq, is_computation_time_required=True,
        )
    return result


# common
def generate_empi_dists_and_calc_estimate(
    qtomography: "StandardQTomography",
    true_object: QOperation,
    num_data: List[int],
    estimator: StandardQTomographyEstimator,
    loss: ProbabilityBasedLossFunction = None,
    loss_option: ProbabilityBasedLossFunctionOption = None,
    algo: MinimizationAlgorithm = None,
    algo_option: MinimizationAlgorithmOption = None,
    iteration: Optional[int] = None,
) -> Union[
    StandardQTomographyEstimationResult, List[StandardQTomographyEstimationResult],
]:

    if iteration is None:
        result = _generate_empi_dists_and_calc_estimate(
            qtomography,
            true_object,
            num_data,
            estimator,
            loss=loss,
            loss_option=loss_option,
            algo=algo,
            algo_option=algo_option,
        )
        return result
    else:
        results = []
        for _ in tqdm(range(iteration)):
            result = _generate_empi_dists_and_calc_estimate(
                qtomography,
                true_object,
                num_data,
                estimator,
                loss=loss,
                loss_option=loss_option,
                algo=algo,
                algo_option=algo_option,
            )
            results.append(result)
        return results
