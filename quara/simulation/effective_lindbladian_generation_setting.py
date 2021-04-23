from typing import List, Union, Tuple

from quara.objects.effective_lindbladian import EffectiveLindbladian
from quara.objects.qoperation import QOperation
from quara.objects.qoperation_typical import generate_effective_lindbladian_object
from quara.simulation.generation_setting import QOperationGenerationSetting


class EffectiveLindbladianGenerationSetting(QOperationGenerationSetting):
    def __init__(
        self,
        c_sys: "CompositeSystem",
        qoperation_base: Union[QOperation, Tuple[str]],
        lindbladian_base: Union[EffectiveLindbladian, str],
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
        ids: List[int], Optional
            This is a list of elmental system's ids, by default None.
            To be given for specific asymmetric multi-partite gates
            For example, in the case of gate_name = 'cx', id[0] is for the id of the control qubit and id[1] is for the id of the target qubit.

        Raises
        ------
        TypeError
            If the type of argument `lindbladian_base` is not EffectiveLindbladian or str.
        """
        super().__init__(c_sys, qoperation_base, ids=ids)

        if isinstance(lindbladian_base, EffectiveLindbladian):
            self._lindbladian_base = lindbladian_base
        elif type(lindbladian_base) == str:
            ids = [e.name for e in c_sys.elemental_systems]
            self._lindbladian_base = generate_effective_lindbladian_object(
                gate_name=lindbladian_base,
                object_name="effective_lindbladian",
                ids=ids,
                c_sys=c_sys,
            )
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
