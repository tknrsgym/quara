from typing import Union

from quara.data_analysis.generation_setting import QOperationGenerationSetting
from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.qoperation import QOperation


class EffectiveLindbladianGenerationSetting(QOperationGenerationSetting):
    def __init__(
        self,
        c_sys: "CompositeSystem",
        qoperation_base: Union[QOperation, str],
        lindbladian_base: Union[EffectiveLindbladian, str],
    ) -> None:
        super().__init__(c_sys, qoperation_base)

        if isinstance(lindbladian_base, EffectiveLindbladian):
            self._lindbladian_base = lindbladian_base
        elif type(lindbladian_base) == str:
            # TODO
            raise NotImplementedError()
        else:
            message = f"`lindbladian_base` must be EffectiveLindbladian or str, not {type(lindbladian_base)}"
            raise TypeError(message)

    @property
    def lindbladian_base(self) -> EffectiveLindbladian:
        return self._lindbladian_base
