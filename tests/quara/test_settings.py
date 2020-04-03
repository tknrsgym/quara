from quara.settings import Settings


class TestSettings:
    def test_atol_getter(self):
        # Arrange & Act
        actual = Settings.get_atol()

        # Assert
        expected = 1e-13
        assert actual == expected

    def test_atol_setter(self):
        # Arrange & Act
        Settings.set_atol(1)
        actual = Settings.get_atol()

        # Assert
        expected = 1
        assert actual == expected
