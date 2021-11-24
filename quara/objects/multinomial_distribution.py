from functools import reduce
from operator import mul
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import multinomial

from quara.math.probability import validate_prob_dist
from quara.utils.index_util import index_serial_from_index_multi_dimensional
from quara.utils.number_util import to_stream


class MultinomialDistribution:
    def __init__(
        self, ps: np.ndarray, shape: Tuple[int] = None, eps_zero: float = None
    ):
        """Constructor

        Parameters
        ----------
        ps : np.ndarray
            the probability distribution of multinomial distribution.
        shape : Tuple[int], optional
            the shape of multinomial distribution, by default None.
            if shape is None, set len(ps).
        eps_zero : float, optional
            threshold to determine probabilities as zero, by default 1e-8

        Raises
        ------
        ValueError
            some elements of ps are negative numbers.
        ValueError
            the sum of ps is not 1.
        ValueError
            the size of ps and shape do not match.
        """
        if type(ps) == list:
            ps = np.array(ps)

        # validation about probabiity
        validate_prob_dist(ps, validate_sum=False)

        if shape is None:
            self._shape = (len(ps),)
        else:
            # validation about data size
            if len(ps) != reduce(mul, shape):
                raise ValueError(
                    f"the size of ps({len(ps)}) and shape({shape}) do not match."
                )
            self._shape = shape

        self._ps = ps
        self._eps_zero = eps_zero if eps_zero else 1e-8

        # adjust probability distribution
        self._is_zero_dist = True
        has_zero = False
        for index, prob in enumerate(self.ps):
            if prob < self.eps_zero:
                self._ps[index] = 0.0
                has_zero = True
            else:
                self._is_zero_dist = False

        # normalize when is_zero_dist = False and ps has zero
        if self.is_zero_dist == False and has_zero == True:
            self._ps = self._ps / np.sum(self._ps)

        # validation about probabiity
        if self.is_zero_dist == False:
            validate_prob_dist(self.ps, validate_sum=True)

    @property  # read only
    def ps(self) -> np.ndarray:
        """returns ps.

        Returns
        -------
        np.ndarray
            the probability distribution of multinomial distribution.
        """
        return self._ps

    @property  # read only
    def shape(self) -> Tuple[int]:
        """returns shape.

        Returns
        -------
        Tuple[int]
            the shape of multinomial distribution.
        """
        return self._shape

    @property  # read only
    def eps_zero(self) -> float:
        """returns threshold to determine probabilities as zero.

        Returns
        -------
        float
            threshold to determine probabilities as zero.
        """
        return self._eps_zero

    @property  # read only
    def is_zero_dist(self) -> bool:
        """returns whether all probabilities are zero.

        Returns
        -------
        bool
            whether all probabilities are zero.
        """
        return self._is_zero_dist

    def __getitem__(self, idx) -> np.float64:
        # Working in progress
        if type(idx) == int:
            # One-dimensional access
            # ex) prob_dist[0]
            return self._ps[idx]
        elif type(idx) == tuple:
            # Multidimensional access
            # ex) prob_dist[(0, 1)]
            serial_index = index_serial_from_index_multi_dimensional(self._shape, idx)
            return self._ps[serial_index]
        else:
            raise TypeError("unsupported type {type(idx)}")

    def __str__(self):
        desc = f"shape = {self.shape}\n"
        desc += f"ps = {self._ps}"
        return desc

    def marginalize(
        self, outcome_indices_remain: List[int]
    ) -> "MultinomialDistribution":
        """marginalize MultinomialDistribution.

        Parameters
        ----------
        outcome_indices_remain : List[int]
            calculate the marginal probability of variable ``outcome_indices_remain``.

        Returns
        -------
        MultinomialDistribution
            marginalized MultinomialDistribution.

        Raises
        ------
        ValueError
            some elements of outcome_indices_remain are out of range.
        """
        # axis of marginalization
        axis = set(range(len(self.shape)))
        for index in outcome_indices_remain:
            if index < 0 or len(self.shape) <= index:
                raise ValueError(
                    f"some elements of outcome_indices_remain are out of range({index})."
                )
            axis.remove(index)
        axis = tuple(axis)

        # marginalize by np.sum
        marginalized = np.sum(self.ps.reshape(self.shape), axis=axis)
        new_dist = MultinomialDistribution(marginalized.flatten(), marginalized.shape)

        return new_dist

    def conditionalize(
        self,
        conditional_variable_indices: List[int],
        conditional_variable_values: List[int],
    ) -> "MultinomialDistribution":
        """conditionalize MultinomialDistribution.

        Parameters
        ----------
        conditional_variable_indices : List[int]
            indices of the given event of the conditional probability.
        conditional_variable_values : List[int]
            values of the given event of the conditional probability.

        Returns
        -------
        MultinomialDistribution
            conditionalized MultinomialDistribution.

        Raises
        ------
        ValueError
            length of conditional_variable_indices does not equal conditional_variable_values.
        ValueError
            conditional_variable_indices has negative numbers.
        ValueError
            conditional_variable_values has negative numbers.
        """
        ### validate
        if len(conditional_variable_indices) != len(conditional_variable_values):
            raise ValueError(
                "length of conditional_variable_indices must equal conditional_variable_values."
                + f" length of conditional_variable_indices is {len(conditional_variable_indices)}."
                + f" length of conditional_variable_values is {len(conditional_variable_values)}."
            )

        for val in conditional_variable_indices:
            if val < 0:
                raise ValueError(
                    "conditional_variable_indices consists of non-negative numbers."
                    + f" conditional_variable_indices={conditional_variable_indices}"
                )

        for val in conditional_variable_values:
            if val < 0:
                raise ValueError(
                    "conditional_variable_values consists of non-negative numbers."
                    + f" conditional_variable_values={conditional_variable_values}"
                )

        ### calc new ps
        # to extract specific columns from old ps, calculate ixgrid of numpy.
        num_variable = len(self.shape)
        ix_args = []
        for num_var in self.shape:
            arg = [True] * num_var
            ix_args.append(arg)

        for var_index, var_value in zip(
            conditional_variable_indices, conditional_variable_values
        ):
            arg = [False] * self.shape[var_index]
            arg[var_value] = True
            ix_args[var_index] = arg
        ixgrid = np.ix_(*ix_args)

        # extract specific columns from old ps
        new_ps = self.ps.reshape(self.shape)[ixgrid]
        new_ps = new_ps.flatten() / np.sum(new_ps)

        ### calc new shape
        new_shape_indices = [True] * num_variable
        for var_index in conditional_variable_indices:
            new_shape_indices[var_index] = False
        new_shape = np.array(self.shape)[new_shape_indices]
        new_shape = tuple(new_shape)

        new_dist = MultinomialDistribution(new_ps, new_shape)
        return new_dist

    def execute_random_sampling(
        self,
        num: int,
        size: int,
        random_generator: Union[int, np.random.Generator] = None,
    ) -> List[np.ndarray]:
        """execute random sampling.

        Parameters
        ----------
        num : int
            number of trials
        size : int
            size of trials. this size equals the length of the list returned by this function.
        random_generator : Union[int, np.random.Generator], optional

            - If the type is int, generates Generator with seed `seed_or_generator` and returned generated random.Generator.
            - If the type is Generator, returns Generator.
            - If argument is None, returns np.random.
            - Default value is None.

        Returns
        -------
        List[np.ndarray]
            list of random sampling.
        """
        stream = to_stream(random_generator)
        samplings = list(multinomial.rvs(num, self.ps, size=size, random_state=stream))
        return samplings
