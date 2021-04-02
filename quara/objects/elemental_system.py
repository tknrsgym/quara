from quara.objects import matrix_basis
from quara.objects.matrix_basis import MatrixBasis, get_comp_basis

import numpy as np


class ElementalSystem:
    """Class for representing an individual quantum system."""

    def __init__(self, name: int, basis: MatrixBasis):
        """Constructor

        Parameters
        ----------
        name : int
            The name of quantum system.
        basis : MatrixBasis
            The basis of quantum system.

        Raises
        ------
        TypeError
            Type of ``name`` must be int.
        """

        # Validate
        if type(name) != int:
            raise TypeError("Type of 'name' must be int.")

        # Set
        self._name: int = name

        # Without system _ id,
        # it is possible to determine if the instances are the same (not the same value)
        # by using ``is``.
        # But just in case, implement ``system_id``.
        self._system_id: int = id(self)
        self._dim: int = basis.dim
        self._basis: MatrixBasis = basis
        self._is_hermitian = self._basis.is_hermitian()
        self._is_orthonormal_hermitian_0thprop_identity = (
            self._basis.is_normal()
            and self._basis.is_orthogonal()
            and self._is_hermitian
            and self._basis.is_0thpropI()
        )

    @property
    def name(self) -> int:  # read only
        """Property to get the name of a quantum system.

        Returns
        -------
        int
            The name of a quantum system.
        """
        return self._name

    @property
    def system_id(self) -> int:  # read only
        """Property to get the system ID of a quantum system.

        Returns
        -------
        int
            The system ID of a quantum system.
            Although ``name`` can be specified by a user,
            the system ID is assigned automatically.
            Now, ``system_id`` is the same as the result of passing an instance to ``id()``.
        """
        return self._system_id

    @property
    def dim(self) -> int:  # read only
        """Property to get the dimension of the basis.

        Returns
        -------
        int
            The dimension of the basis.
        """
        return self._dim

    @property
    def comp_basis(self) -> MatrixBasis:  # read only
        return get_comp_basis(self._dim)

    @property
    def basis(self) -> MatrixBasis:  # read only
        return self._basis

    @property
    def is_orthonormal_hermitian_0thprop_identity(self) -> bool:  # read only
        """Property to get whether basis is orthonormal, hermitian and 0th prop identity.

        Returns
        -------
        bool
            whether basis is orthonormal, hermitian and 0th prop I.
        """
        return self._is_orthonormal_hermitian_0thprop_identity

    @property
    def is_hermitian(self) -> bool:  # read only
        return self._is_hermitian

    def __str__(self):
        desc = f"name: {self.name} \n"
        desc += f"system_id: {self.system_id} \n"
        desc += f"basis: {self.basis}"
        return desc

    def __repr__(self):
        return f"{self.__class__.__name__}(name={repr(self.name)}, \
            basis={repr(self.basis)})"
