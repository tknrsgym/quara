from functools import reduce
from operator import mul
from typing import List, Tuple, Union

import numpy as np
from scipy.stats import multinomial

from quara.math.probability import validate_prob_dist
from quara.utils.index_util import index_serial_from_index_tuple
from quara.utils.number_util import to_stream


class MultinomialDistribution:
    def __init__(self, ps: np.ndarray, shape: Tuple[int]):
        """Constructor

        Parameters
        ----------
        ps : np.ndarray
            the probability distribution of multinomial distribution.
        shape : Tuple[int], optional
            the shape of multinomial distribution.

        Raises
        ------
        ValueError
            some elements of ps are negative numbers.
        ValueError
            the sum of ps is not 1.
        ValueError
            the size of ps and shape do not match.
        """
        # validation about probabiity
        validate_prob_dist(ps)

        # validation about data size
        if len(ps) != reduce(mul, shape):
            raise ValueError(
                f"the size of ps({len(ps)}) and shape({shape}) do not match."
            )

        self._ps = ps
        self._shape = shape

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

    def __getitem__(self, idx) -> np.float64:
        # Working in progress
        if type(idx) == int:
            # One-dimensional access
            # ex) prob_dist[0]
            return self._ps[idx]
        elif type(idx) == tuple:
            # Multidimensional access
            # ex) prob_dist[(0, 1)]
            serial_index = index_serial_from_index_tuple(self._shape, idx)
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

    def execute_random_sampling(
        self,
        num: int,
        size: int,
        random_state: Union[int, np.random.RandomState] = None,
    ) -> List[np.ndarray]:
        """execute random sampling.

        Parameters
        ----------
        num : int
            number of trials
        size : int
            size of trials. this size equals the length of the list returned by this function.
        random_state : Union[int, np.random.RandomState], optional

            - If the type is int, generates RandomState with seed `seed_or_stream` and returned generated RandomState.
            - If the type is RandomState, returns RandomState.
            - If argument is None, returns np.random.
            - Default value is None.

        Returns
        -------
        List[np.ndarray]
            list of random sampling.
        """
        stream = to_stream(random_state)
        samplings = list(multinomial.rvs(num, self.ps, size=size, random_state=stream))
        return samplings
