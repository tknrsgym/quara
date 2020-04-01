class Settings:
    atol = 1e-13

    @classmethod
    def get_atol(cls) -> float:
        """returns global setting of ``atol``.

        default value is 1e-13.
        
        Returns
        -------
        float
            global setting of ``atol``.
        """
        return cls.atol

    @classmethod
    def set_stol(cls, atol: float):
        """updates global setting of ``atol``.
        
        Parameters
        ----------
        atol : float
            global setting of ``atol``.
        """
        cls.atol = atol
