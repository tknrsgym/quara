from abc import abstractmethod
from typing import List, Tuple

import numpy as np

from quara.objects.qoperation import QOperation
from quara.objects.qoperations import SetQOperations
from quara.qcircuit.experiment import Experiment


class QTomography:
    def __init__(
        self, experiment: Experiment, set_qoperations: SetQOperations,
    ):
        """initialize quantum tomography class.

        To inherit from this class, set the following instance variables in the constructor of the subclass.

        - ``_num_variables``: sum of the number of all variables.

        Parameters
        ----------
        experiment : Experiment
            Experiment class used in quantum tomography.
        set_qoperations : SetQOperations
            SetQOperations class used in quantum tomography.
        """
        self._experiment = experiment
        self._num_schedules = len(self._experiment.schedules)
        self._set_qoperations = set_qoperations

        # validate ElementalSystem of Experiment
        for state in self._experiment.states:
            if (
                not state is None
                and not state.composite_system.is_orthonormal_hermitian_0thpropI
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(state)} is not so."
                )
        for gate in self._experiment.gates:
            if (
                not gate is None
                and not gate.composite_system.is_orthonormal_hermitian_0thpropI
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(gate)} is not so."
                )
        for povm in self._experiment.povms:
            if (
                not povm is None
                and not povm.composite_system.is_orthonormal_hermitian_0thpropI
            ):
                raise ValueError(
                    f"all ElementalSystem of Experiment must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(povm)} is not so."
                )

        # validate ElementalSystem of SetQOperations
        for state in self._set_qoperations.states:
            if (
                not state is None
                and not state.composite_system.is_orthonormal_hermitian_0thpropI
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(state)} is not so."
                )
        for gate in self._set_qoperations.gates:
            if (
                not gate is None
                and not gate.composite_system.is_orthonormal_hermitian_0thpropI
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(gate)} is not so."
                )
        for povm in self._set_qoperations.povms:
            if (
                not povm is None
                and not povm.composite_system.is_orthonormal_hermitian_0thpropI
            ):
                raise ValueError(
                    f"all ElementalSystem of SetQOperations must be orthonormal, hermitian and 0th prop I. the ElementalSystem of {str(povm)} is not so."
                )

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
    def calc_prob_dist(self) -> List[np.array]:
        """calculates a probability distribution.

        this function must be implemented in the subclass.

        Returns
        -------
        List[np.array]
            a probability distribution.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def calc_prob_dists(self) -> List[List[np.array]]:
        """calculates probability distributions.

        this function must be implemented in the subclass.

        Returns
        -------
        List[List[np.array]]
            probability distributions.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_dataset(
        self, data_nums: List[int], seeds: List[int] = None,
    ) -> List[List[np.array]]:
        """Run all the schedules to caluclate the probability distribution and generate random data.

        this function must be implemented in the subclass.

        Parameters
        ----------
        data_nums : List[int]
            A list of the number of data to be generated in each schedule. This parameter should be a list of non-negative integers.
        seeds : List[int], optional
            A list of the seeds to be used in each schedule, by default None

        Returns
        -------
        List[List[np.array]]
            Generated dataset.

        Raises
        ------
        NotImplementedError
            this function does not be implemented in the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def generate_empi_dists(
        self, qoperation: QOperation, num_sum: int, seed: int = None
    ) -> List[Tuple[int, np.array]]:
        """Generate empirical distributions using the data generated from probability distributions of all schedules.

        this function must be implemented in the subclass.

        Parameters
        ----------
        qoperation : QOperation
            QOperation to use to generate the experience distributions.
        num_sum : int
            the number of data to use to generate the experience distributions for each schedule.
        seed : int, optional
            the seed, by default None

        Returns
        -------
        List[Tuple[int, np.array]]
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
