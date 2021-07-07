from typing import Union, Tuple, List
from abc import abstractmethod
import dataclasses

from quara.objects.qoperation import QOperation
from quara.objects.qoperation_typical import generate_qoperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate


class QOperationGenerationSetting:
    def __init__(
        self,
        c_sys: "CompositeSystem",
        qoperation_base: Union[QOperation, Tuple[str]],
        ids: List[int] = None,
        is_seed_or_stream_required: bool = False,
    ) -> None:
        self._composite_system = c_sys
        self._is_seed_or_stream_required = is_seed_or_stream_required

        type_error_message = "Type of 'qoperation_base' must be QOperation or tuple of length 2 containing the string (`('state', 'z0')`, etc.), "
        type_error_message += f"not {type(qoperation_base)}"
        if isinstance(qoperation_base, QOperation):
            self._qoperation_base = qoperation_base
        elif type(qoperation_base) == tuple:
            # Validation
            if len(qoperation_base) != 2:
                raise TypeError(type_error_message)
            else:
                for item in qoperation_base:
                    if type(item) != str:
                        raise TypeError(type_error_message)
            # Generate QOperation object.
            qoperation_type = qoperation_base[0]
            qoperation_name = qoperation_base[1]
            self._qoperation_base = generate_qoperation(
                mode=qoperation_type, name=qoperation_name, c_sys=c_sys, ids=ids
            )
        else:
            raise TypeError(type_error_message)

    @property
    def composite_system(self):
        return self._composite_system

    @property
    def qoperation_base(self):
        return self._qoperation_base

    @property
    def is_seed_or_stream_required(self):
        return self._is_seed_or_stream_required

    def generate(self):
        if type(self.qoperation_base) == State:
            return self.generate_state()

        if type(self.qoperation_base) == Povm:
            return self.generate_povm()

        if type(self.qoperation_base) == Gate:
            return self.generate_gate()

        raise NotImplementedError()

    @abstractmethod
    def generate_state(self):
        raise NotImplementedError()

    @abstractmethod
    def generate_povm(self):
        raise NotImplementedError()

    @abstractmethod
    def generate_gate(self):
        raise NotImplementedError()


@dataclasses.dataclass
class QOperationGenerationSettings:
    true_setting: QOperationGenerationSetting
    tester_settings: List[QOperationGenerationSetting]
