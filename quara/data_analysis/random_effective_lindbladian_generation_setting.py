from abc import abstractmethod
from functools import reduce
from operator import add
from typing import Tuple, Union

import numpy as np

from quara.data_analysis.effective_lindbladian_generation_setting import (
    EffectiveLindbladianGenerationSetting,
)
from quara.objects.effective_lindbladian import (
    EffectiveLindbladian,
    generate_effective_lindbladian_from_h,
)
from quara.objects.gate import Gate
from quara.objects.operators import composite
from quara.objects.povm import Povm
from quara.objects.qoperation import QOperation
from quara.objects.state import State


class RandomEffectiveLindbladianGenerationSetting(
    EffectiveLindbladianGenerationSetting
):
    def __init__(
        self,
        c_sys: "CompositeSystem",
        qoperation_base: Union[QOperation, str],
        lindbladian_base: Union[EffectiveLindbladian, str],
        strength_h_part: float,
        strength_k_part: float,
    ) -> None:
        super().__init__(c_sys, qoperation_base, lindbladian_base)
        self._strength_h_part = strength_h_part
        self._strength_k_part = strength_k_part

    @property
    def strength_h_part(self) -> float:
        return self._strength_h_part

    @property
    def strength_k_part(self) -> float:
        return self._strength_k_part

    def _generate_random_variables(self, strength: float):
        dim = self.composite_system.dim
        random_variables = np.random.randn(dim ** 2 - 1)
        normalized_factor = 1 / np.sqrt(np.sum(random_variables ** 2))
        random_variables = strength * normalized_factor * random_variables
        return random_variables

    def generate_random_effective_lindbladian_h_part(self) -> EffectiveLindbladian:
        # generate randum variables
        random_variables = self._generate_random_variables(self.strength_h_part)

        # generate EffectiveLindbladian
        basis = self.composite_system.basis()
        terms = []
        for index, h_alpha in enumerate(random_variables):
            terms.append(h_alpha * basis[index + 1])

        delta_h_mat = reduce(add, terms)
        random_el = generate_effective_lindbladian_from_h(
            self.composite_system, delta_h_mat, is_physicality_required=False
        )

        # calculate composite_qoperations
        qoperation = composite(random_el, self.qoperation_base)

        return qoperation, random_variables

    def generate_random_effective_lindbladian_j_part(self) -> EffectiveLindbladian:
        pass

    def generate_random_effective_lindbladian_k_part(self) -> EffectiveLindbladian:
        pass

    def generate_random_effective_lindbladian_d_part(self) -> EffectiveLindbladian:
        pass

    def generate_random_effective_lindbladian(self) -> EffectiveLindbladian:
        h_part, h_part_rv = self.generate_random_effective_lindbladian_h_part()
        (
            d_part,
            k_part_rv,
            random_unitary,
        ) = self.generate_random_effective_lindbladian_d_part()
        el = h_part + d_part
        return el, h_part_rv, k_part_rv, random_unitary

    def generate_state(self) -> State:
        (
            el,
            h_part_rv,
            k_part_rv,
            random_unitary,
        ) = self.generate_random_effective_lindbladian()
        new_object = composite(el, self.qoperation_base)
        return (
            new_object,
            h_part_rv,
            k_part_rv,
            random_unitary,
        )

    def generate_gate(self) -> Gate:
        (
            el,
            h_part_rv,
            k_part_rv,
            random_unitary,
        ) = self.generate_random_effective_lindbladian()
        new_object = composite(el, self.qoperation_base)
        return (
            new_object,
            h_part_rv,
            k_part_rv,
            random_unitary,
        )

    def generate_povm(self) -> Povm:
        (
            el,
            h_part_rv,
            k_part_rv,
            random_unitary,
        ) = self.generate_random_effective_lindbladian()
        new_object = composite(self.qoperation_base, el)
        return (
            new_object,
            h_part_rv,
            k_part_rv,
            random_unitary,
        )
