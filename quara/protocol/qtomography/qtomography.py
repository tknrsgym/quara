from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.state import State
from quara.objects.povm import Povm
from quara.objects.gate import Gate
from quara.objects.mprocess import MProcess
from quara.objects.qoperations import SetQOperations
from quara.qcircuit.experiment import Experiment


class QTomography:
    def __init__(
        self,
        experiment: Experiment,
        set_qoperations: SetQOperations,
    ):
        """initialize quantum tomography class.

        To inherit from this class, set the following instance variables in the constructor of the subclass.

        - ``_num_variables``: sum of the number of all variables.

        Parameters
        ----------
        experiment : Experiment
            Experiment class used in quantum tomography, which is supposed to have tester objects.
        set_qoperations : SetQOperations
            SetQOperations class used in quantum tomography, which is supposed to have conditions of true object.
        """
        self._experiment = experiment
        self._num_schedules = len(self._experiment.schedules)
        self._set_qoperations = set_qoperations

        # validate ElementalSystem of Experiment
        for state in self._experiment.states:
            if (
                not state is None
                and not state.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(state)} is not so."
                )
            # whether entries of vec of State are real numbers
            if not state is None and state.vec.dtype != np.float64:
                raise ValueError(
                    f"entries of vec of State must be real numbers. dtype of vec is {state.vec.dtype}"
                )
        for gate in self._experiment.gates:
            if (
                not gate is None
                and not gate.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(gate)} is not so."
                )
            # whether entries of HS representation of Gate are real numbers
            if not gate is None and gate.hs.dtype != np.float64:
                raise ValueError(
                    f"entries of HS representation of Gate must be real numbers. dtype of HS is {gate.hs.dtype}"
                )
        for povm in self._experiment.povms:
            if (
                not povm is None
                and not povm.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(povm)} is not so."
                )
            # whether entries of vecs of Povm are real numbers
            if not povm is None:
                for vec in povm.vecs:
                    if vec.dtype != np.float64:
                        raise ValueError(
                            f"entries of all vecs of Povm must be real numbers. some dtype of vecs are {vec.dtype}"
                        )

        for mprocess in self._experiment.mprocesses:
            if (
                not mprocess is None
                and not mprocess.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(mprocess)} is not so."
                )
            # whether entries of vecs of Povm are real numbers
            if not mprocess is None:
                for hs in mprocess.hss:
                    if hs.dtype != np.float64:
                        raise ValueError(
                            f"entries of all hss of MProcess must be real numbers. some dtype of vecs are {hs.dtype}"
                        )

        # validate ElementalSystem of SetQOperations
        for state in self._set_qoperations.states:
            if (
                not state is None
                and not state.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(state)} is not so."
                )
        for gate in self._set_qoperations.gates:
            if (
                not gate is None
                and not gate.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(gate)} is not so."
                )
        for povm in self._set_qoperations.povms:
            if (
                not povm is None
                and not povm.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(povm)} is not so."
                )

        for mprocess in self._set_qoperations.mprocesses:
            if (
                not mprocess is None
                and not mprocess.composite_system.is_orthonormal_hermitian_0thprop_identity
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(mprocess)} is not so."
                )

    @property
    def experiment(self) -> Experiment:
        """returns Experiment class.

        Returns
        -------
        Experiment
            Experiment class.
        """
        return self._experiment

    @property
    def set_qoperations(self) -> SetQOperations:
        """returns SetQOperations class.

        Returns
        -------
        SetQOperations
            SetQOperations class.
        """
        return self._set_qoperations

    @property
    def num_schedules(self) -> int:
        """returns number of schedules.

        Returns
        -------
        int
            number of schedules.
        """
        return self._num_schedules

    @property
    def num_variables(self) -> int:
        """returns sum of the number of all variables.

        Returns
        -------
        int
            sum of the number of all variables.
        """
        return self._num_variables

    @property
    def states(self) -> List[State]:
        return self._experiment.states

    @property
    def povms(self) -> List[Povm]:
        return self._experiment.povms

    @property
    def gates(self) -> List[Gate]:
        return self._experiment.gates

    @property
    def mprocesses(self) -> List[MProcess]:
        return self._experiment.mprocesses

    @abstractmethod
    def _testers(self) -> QOperation:
        raise NotImplementedError()

    @property
    def testers(self) -> List[QOperation]:
        """returns tester objects.

        Returns
        -------
        List[QOperation]
            tester objects.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        return self._testers()

    def reset_seed(self, seed: int = None) -> None:
        """reset new seed.

        if `seed` is None, reset by seed which Experiment already has.

        Parameters
        ----------
        seed : int, optional
            new seed, None by default.
        """
        if seed:
            self._experiment.reset_seed_data(seed)
        else:
            self._experiment.reset_seed_data(self._experiment.seed_data)

    @abstractmethod
    def is_valid_experiment(self) -> bool:
        """returns whether the experiment is valid.

        this function must be implemented in the subclass.

        Returns
        -------
        bool
            whether the experiment is valid.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_prob_dist(self, qope: QOperation, schedule_index: int) -> List[float]:
        """calculates a probability distribution.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate a probability distribution.
        schedule_index : int
            schedule index.

        Returns
        -------
        List[np.ndarray]
            a probability distribution.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_prob_dists(self, qope: QOperation) -> List[List[float]]:
        """calculates probability distributions.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qope : QOperation
            QOperation to calculate probability distributions.

        Returns
        -------
        List[List[np.ndarray]]
            probability distributions.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_dataset(self, data_nums: List[int]) -> List[List[np.ndarray]]:
        """Run all the schedules to caluclate the probability distribution and generate random data.

        this function must be implemented in the subclass.

        Parameters
        ----------
        data_nums : List[int]
            A list of the number of data to be generated in each schedule. This parameter should be a list of non-negative integers.

        Returns
        -------
        List[List[np.ndarray]]
            Generated dataset.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_empi_dists(
        self, qoperation: QOperation, num_sum: int
    ) -> List[Tuple[int, np.ndarray]]:
        """Generate empirical distributions using the data generated from probability distributions of all schedules.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qoperation : QOperation
            QOperation to use to generate the experience distributions.
        num_sum : int
            the number of data to use to generate the experience distributions for each schedule.

        Returns
        -------
        List[Tuple[int, np.ndarray]]
            A list of tuples for the number of data and experience distributions for each schedules.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def func_prob_dist(self):
        raise NotImplementedError()

    @abstractmethod
    def func_prob_dists(self):
        raise NotImplementedError()
