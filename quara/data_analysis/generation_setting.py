from typing import Type, Union, Tuple
from abc import abstractmethod

from quara.objects.qoperation import QOperation
from quara.objects.qoperation_typical import generate_qoperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate, get_depolarizing_channel
from quara.objects.operators import composite


class QOperationGenerationSetting:
    def __init__(
        self, c_sys: "CompositeSystem", qoperation_base: Union[QOperation, Tuple[str]]
    ) -> None:
        self._composite_system = c_sys

        type_error_message = "Type of 'qooeration_base' must be QOperation or tuple of length 2 containing the string, "
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
                mode=qoperation_type, name=qoperation_name, c_sys=c_sys
            )
        else:
            raise TypeError(type_error_message)

    @property
    def composite_system(self):
        return self._composite_system

    @property
    def qoperation_base(self):
        return self._qoperation_base

    def generate(self):
        if type(self.qoperation_base) == State:
            return self.generate_state()

        if type(self.qoperation_base) == Povm:
            return self.generate_povm()

        if type(self.qoperation_base) == Gate:
            return self.generate_gate()

        # TODO: imprement MProcess
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


class DepolarizedQOperationGenerationSetting(QOperationGenerationSetting):
    def __init__(
        self, c_sys, qoperation_base: Union[QOperation, Tuple[str]], error_rate: float
    ) -> None:
        if not (0 <= error_rate <= 1):
            message = "`error_rate` must be between 0 and 1."
            raise ValueError(message)

        super().__init__(c_sys=c_sys, qoperation_base=qoperation_base)
        self._error_rate = error_rate

    @property
    def error_rate(self):
        return self._error_rate

    def generate_state(self):
        dp = get_depolarizing_channel(
            p=self.error_rate, c_sys=self.qoperation_base.composite_system
        )
        new_object = composite(dp, self.qoperation_base)
        return new_object

    def generate_povm(self):
        dp = get_depolarizing_channel(
            p=self.error_rate, c_sys=self.qoperation_base.composite_system
        )
        new_object = composite(self.qoperation_base, dp)
        return new_object

    def generate_gate(self):
        dp = get_depolarizing_channel(
            p=self.error_rate, c_sys=self.qoperation_base.composite_system
        )
        new_object = composite(dp, self.qoperation_base)
        return new_object

