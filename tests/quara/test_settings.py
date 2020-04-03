import pytest

from quara.settings import Settings


@pytest.fixture()
def fixture_atol():
    first_default_atol = Settings.get_atol()
    yield
    Settings.set_atol(first_default_atol)


class TestSettings:
    def test_atol_getter(self):
        # Arrange & Act
        actual = Settings.get_atol()

        # Assert
        expected = 1e-13
        assert actual == expected

    def test_atol_setter(self, fixture_atol):
        # Arrange & Act
        Settings.set_atol(1.0)
        actual = Settings.get_atol()

        # Assert
        expected = 1.0
        assert actual == expected

    def test_atol_setter_validation_ng(self):
        # Arrange & Act & Assert
        with pytest.raises(TypeError):
            # TypeError: Type of `atol` must be float.
            Settings.set_atol(1)
