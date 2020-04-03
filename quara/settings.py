class Settings:
    __atol = 1e-13

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
    def set_atol(cls, atol: float):
        """updates global setting of ``atol``.

        Parameters
        ----------
        atol : float
            global setting of ``atol``.
        """
        cls.__atol = atol
