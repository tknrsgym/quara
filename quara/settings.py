"""Module for global settings"""


class Settings:
    __first_default_atol = 1e-13
    __atol = __first_default_atol

    @classmethod
    def get_atol(cls) -> float:
        """returns global setting of ``atol``.

        default value is 1e-13.

        Returns
        -------
        float
            global setting of ``atol``.
        """
        return cls.__atol

    @classmethod
    def set_atol(cls, atol: float) -> None:
        """updates global setting of ``atol``.

        Parameters
        ----------
        atol : float
            global setting of ``atol``.
        """
        if type(atol) != float:
            raise TypeError("Type of `atol` must be float.")
        cls.__atol = atol
