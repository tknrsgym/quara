import time
from typing import List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.protocol.qtomography.standard.standard_qtomography import StandardQTomography
from quara.protocol.qtomography.standard.linear_estimator import (
    LinearEstimator,
    LinearEstimationResult,
)


class ProjectedLinearEstimationResult(LinearEstimationResult):
    def __init__(
        self,
        qtomography: StandardQTomography,
        data,
        estimated_var_sequence: List[np.ndarray],
        computation_times: List[float],
    ):
        super().__init__(
            qtomography,
            data,
            estimated_var_sequence,
            computation_times,
        )


class ProjectedLinearEstimator(LinearEstimator):
    def __init__(
        self,
        mode_proj_order: str = "eq_ineq",
    ):
        super().__init__()

        if not mode_proj_order in ["eq_ineq", "ineq_eq"]:
            raise ValueError(f"unsupported mode_proj_order={mode_proj_order}")
        self._mode_proj_order: str = mode_proj_order

    @property
    def mode_proj_order(self) -> str:
        """returns the order in which the projections are performed.

        Returns
        -------
        str
            the order in which the projections are performed.
        """
        return self._mode_proj_order

    def calc_estimate(
        self,
        qtomography: StandardQTomography,
        empi_dists: List[Tuple[int, np.ndarray]],
        is_computation_time_required: bool = False,
    ) -> LinearEstimationResult:
        """calculates estimate variables.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography_estimator.StandardQTomographyEstimator.calc_estimate`
        """
        result = self.calc_estimate_sequence(
            qtomography,
            [empi_dists],
            is_computation_time_required=is_computation_time_required,
        )

        return result

    def calc_estimate_sequence(
        self,
        qtomography: StandardQTomography,
        empi_dists_sequence: List[List[Tuple[int, np.ndarray]]],
        is_computation_time_required: bool = False,
    ) -> LinearEstimationResult:
        """calculates sequence of estimate variables.

        see :func:`~quara.protocol.qtomography.standard.standard_qtomography_estimator.StandardQTomographyEstimator.calc_estimate_sequence`
        """
        result = super().calc_estimate_sequence(
            qtomography, empi_dists_sequence, is_computation_time_required
        )
        linear_estimates = result.estimated_qoperation_sequence
        linear_computation_times = result.computation_times

        proj_estimated_var_sequence = []
        proj_computation_times = [] if is_computation_time_required else None
        for index, linear_estimate in enumerate(linear_estimates):
            if is_computation_time_required:
                start_time = time.time()

            linear_estimate.set_mode_proj_order(self.mode_proj_order)
            proj_estimate = linear_estimate.calc_proj_physical(
                is_iteration_history=is_computation_time_required
            )

            if is_computation_time_required:
                comp_time = time.time() - start_time
                proj_computation_times.append(
                    linear_computation_times[index] + comp_time
                )
                proj_estimated_var_sequence.append(proj_estimate[0].to_var())
            else:
                proj_estimated_var_sequence.append(proj_estimate.to_var())

        result = ProjectedLinearEstimationResult(
            qtomography,
            empi_dists_sequence,
            proj_estimated_var_sequence,
            proj_computation_times,
        )
        return result
