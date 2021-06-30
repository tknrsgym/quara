from functools import reduce
from operator import add
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import unitary_group

from quara.objects.effective_lindbladian import (
    EffectiveLindbladian,
    _calc_h_part_from_h_mat,
    _calc_k_part_from_k_mat,
    _calc_j_mat_from_k_mat,
    _calc_j_part_from_j_mat,
    _truncate_hs,
)
from quara.objects.gate import Gate, convert_hs
from quara.objects.operators import compose_qoperations
from quara.objects.povm import Povm
from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.simulation.effective_lindbladian_generation_setting import (
    EffectiveLindbladianGenerationSetting,
)


class RandomEffectiveLindbladianGenerationSetting(
    EffectiveLindbladianGenerationSetting
):
    def __init__(
        self,
        c_sys: "CompositeSystem",
        qoperation_base: Union[QOperation, Tuple[str]],
        lindbladian_base: Union[EffectiveLindbladian, str],
        strength_h_part: float,
        strength_k_part: float,
        ids: List[int] = None,
    ) -> None:
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem.
        qoperation_base : Union[QOperation, str]
            QOperation base of the random effective Lindbladian.
        lindbladian_base : Union[EffectiveLindbladian, str]
            effective Lindbladian base of the random effective Lindbladian.
        strength_h_part : float
            the strength of random variables for generating h part.
        strength_k_part : float
            the strength of random variables for generating k part.
        ids: List[int], Optional
            This is a list of elmental system's ids, by default None.
            To be given for specific asymmetric multi-partite gates
            For example, in the case of gate_name = 'cx', id[0] is for the id of the control qubit and id[1] is for the id of the target qubit.

        Raises
        ------
        ValueError
            strength_h_part is not non-negative number.
        ValueError
            strength_k_part is not non-negative number.
        """
        # validation
        if strength_h_part < 0:
            raise ValueError(
                f"strength_h_part must be non-negative number. strength_h_part is {strength_h_part}"
            )
        if strength_k_part < 0:
            raise ValueError(
                f"strength_k_part must be non-negative number. strength_k_part is {strength_k_part}"
            )

        super().__init__(c_sys, qoperation_base, lindbladian_base, ids=ids)
        self._strength_h_part = strength_h_part
        self._strength_k_part = strength_k_part

    @property
    def strength_h_part(self) -> float:
        """returns the strength of random variables for generating h part.

        Returns
        -------
        float
            the strength of random variables for generating h part.
        """
        return self._strength_h_part

    @property
    def strength_k_part(self) -> float:
        """returns the strength of random variables for generating k part.

        Returns
        -------
        float
            the strength of random variables for generating k part.
        """
        return self._strength_k_part

    def _generate_random_variables(self, strength: float):
        dim = self.composite_system.dim
        random_variables = np.random.randn(dim ** 2 - 1)
        normalized_factor = 1 / np.sqrt(np.sum(random_variables ** 2))
        random_vector = strength * normalized_factor * random_variables
        return random_vector, random_variables

    def generate_random_effective_lindbladian_h_part(
        self,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """generates random HS matrix on computational basis of h part of effective Lindbladian.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            tuple of random HS matrix and ramdom variables.
        """
        # generate randum variables
        random_vector, random_variables = self._generate_random_variables(
            self.strength_h_part
        )

        # calc random h mat
        basis = self.composite_system.basis()
        terms = []
        for index, h_alpha in enumerate(random_vector):
            terms.append(h_alpha * basis[index + 1])
        random_h_mat = reduce(add, terms)

        # calc random h part
        random_h_part_cb = _calc_h_part_from_h_mat(random_h_mat)

        return random_h_part_cb, random_variables

    def generate_random_effective_lindbladian_d_part(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """generates random HS matrix on computational basis of d part of effective Lindbladian.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            tuple of random HS matrix, ramdom variables and random unitary matrix.
        """
        # generate randum variables
        random_vector, random_variables = self._generate_random_variables(
            self.strength_k_part
        )
        random_vector = np.abs(random_vector)

        # generate randum variables
        dim = self.composite_system.dim
        random_unitary = unitary_group.rvs(dim ** 2 - 1)

        # calc random k mat
        random_k_mat = random_unitary @ np.diag(random_vector) @ random_unitary.T.conj()

        # calc random d part
        random_j_part_cb = _calc_j_part_from_j_mat(
            _calc_j_mat_from_k_mat(random_k_mat, self.composite_system)
        )
        random_k_part_cb = _calc_k_part_from_k_mat(random_k_mat, self.composite_system)
        random_d_part_cb = random_j_part_cb + random_k_part_cb

        return random_d_part_cb, random_variables, random_unitary

    def generate_random_effective_lindbladian(
        self,
    ) -> Tuple[EffectiveLindbladian, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """generates random effective Lindbladian and returns effective Lindbladian base + random effective Lindbladian.

        Returns
        -------
        Tuple[ EffectiveLindbladian, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ]
            tuple of effective Lindbladian, ramdom variables for h part, ramdom variables for k part, random unitary matrix and random effective Lindbladian.
        """
        (
            random_h_part_cb,
            random_variables_h_part,
        ) = self.generate_random_effective_lindbladian_h_part()
        (
            random_d_part_cb,
            random_variables_k_part,
            random_unitary,
        ) = self.generate_random_effective_lindbladian_d_part()
        random_el_cb = random_h_part_cb + random_d_part_cb

        random_el_gb_tmp = convert_hs(
            random_el_cb,
            self.composite_system.comp_basis(),
            self.composite_system.basis(),
        )
        random_el_gb = _truncate_hs(
            random_el_gb_tmp, self.lindbladian_base.eps_proj_physical
        )

        new_hs = self.lindbladian_base.hs + random_el_gb
        el = EffectiveLindbladian(
            self.composite_system,
            new_hs,
            is_physicality_required=False,
            is_estimation_object=self.lindbladian_base.is_estimation_object,
            on_para_eq_constraint=self.lindbladian_base.on_para_eq_constraint,
            on_algo_eq_constraint=self.lindbladian_base.on_algo_eq_constraint,
            on_algo_ineq_constraint=self.lindbladian_base.on_algo_ineq_constraint,
            mode_proj_order=self.lindbladian_base.mode_proj_order,
            eps_proj_physical=self.lindbladian_base.eps_proj_physical,
        )

        return (
            el,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el_gb,
        )

    def generate_state(
        self,
    ) -> Tuple[State, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """generates random effective Lindbladian and returns state(composition of random effective Lindbladian and qoperation base).

        Returns
        -------
        Tuple[ State, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ]
            tuple of state, ramdom variables for h part, ramdom variables for k part, random unitary matrix and random effective Lindbladian.

        """
        (
            el,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = self.generate_random_effective_lindbladian()
        new_object = compose_qoperations(el.to_gate(), self.qoperation_base)
        return (
            new_object,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        )

    def generate_gate(
        self,
    ) -> Tuple[Gate, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """generates random effective Lindbladian and returns gate(composition of random effective Lindbladian and qoperation base).

        Returns
        -------
        Tuple[ Gate, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ]
            tuple of gate, ramdom variables for h part, ramdom variables for k part, random unitary matrix and random effective Lindbladian.

        """
        (
            el,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = self.generate_random_effective_lindbladian()
        new_object = compose_qoperations(el.to_gate(), self.qoperation_base)
        return (
            new_object,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        )

    def generate_povm(
        self,
    ) -> Tuple[Povm, np.ndarray, np.ndarray, np.ndarray, np.ndarray,]:
        """generates random effective Lindbladian and returns povm(composition of random effective Lindbladian and qoperation base).

        Returns
        -------
        Tuple[ Povm, np.ndarray, np.ndarray, np.ndarray, np.ndarray, ]
            tuple of povm, ramdom variables for h part, ramdom variables for k part, random unitary matrix and random effective Lindbladian.

        """
        (
            el,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        ) = self.generate_random_effective_lindbladian()
        new_object = compose_qoperations(self.qoperation_base, el.to_gate())
        return (
            new_object,
            random_variables_h_part,
            random_variables_k_part,
            random_unitary,
            random_el,
        )
