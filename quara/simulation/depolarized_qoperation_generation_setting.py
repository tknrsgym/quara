from typing import Union, Tuple, List

from quara.objects.qoperation import QOperation
from quara.objects.gate import get_depolarizing_channel
from quara.objects.operators import compose_qoperations
from quara.simulation.generation_setting import QOperationGenerationSetting


class DepolarizedQOperationGenerationSetting(QOperationGenerationSetting):
    def __init__(
        self,
        c_sys,
        qoperation_base: Union[QOperation, Tuple[str]],
        error_rate: float,
        ids: List[int] = None,
    ) -> None:
        if not (0 <= error_rate <= 1):
            message = "`error_rate` must be between 0 and 1."
            raise ValueError(message)

        super().__init__(
            c_sys=c_sys,
            qoperation_base=qoperation_base,
            is_seed_or_stream_required=False,
            ids=ids,
        )
        self._error_rate = error_rate

    @property
    def error_rate(self) -> float:
        return self._error_rate

    def generate_state(self) -> "State":
        dp = get_depolarizing_channel(
            p=self.error_rate, c_sys=self.qoperation_base.composite_system
        )
        new_object = compose_qoperations(dp, self.qoperation_base)
        return new_object

    def generate_povm(self) -> "Povm":
        dp = get_depolarizing_channel(
            p=self.error_rate, c_sys=self.qoperation_base.composite_system
        )
        new_object = compose_qoperations(self.qoperation_base, dp)
        return new_object

    def generate_gate(self) -> "Gate":
        dp = get_depolarizing_channel(
            p=self.error_rate, c_sys=self.qoperation_base.composite_system
        )
        new_object = compose_qoperations(dp, self.qoperation_base)
        return new_object
