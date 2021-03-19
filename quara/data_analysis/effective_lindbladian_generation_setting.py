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
        """Constructor

        Parameters
        ----------
        c_sys : CompositeSystem
            CompositeSystem.
        qoperation_base : Union[QOperation, str]
            QOperation base of the random effective Lindbladian.
        lindbladian_base : Union[EffectiveLindbladian, str]
            effective Lindbladian base of the random effective Lindbladian.

        Raises
        ------
        TypeError
            If the type of argument `lindbladian_base` is not EffectiveLindbladian or str.
        """
        super().__init__(c_sys, qoperation_base)

        if isinstance(lindbladian_base, EffectiveLindbladian):
            self._lindbladian_base = lindbladian_base
        elif type(lindbladian_base) == str:
            # TODO
            raise NotImplementedError()
        else:
            message = f"type of `lindbladian_base` must be EffectiveLindbladian or str, type of `lindbladian_base` is {type(lindbladian_base)}"
            raise TypeError(message)

    @property
    def lindbladian_base(self) -> EffectiveLindbladian:
        """returns effective Lindbladian base of the random effective Lindbladian.

        Returns
        -------
        EffectiveLindbladian
            effective Lindbladian base of the random effective Lindbladian.
        """
        return self._lindbladian_base