from quara.objects.state_ensemble_typical import (
    generate_state_ensemble_object_from_state_ensemble_name_object_name,
)
from quara.objects.composite_system_typical import generate_composite_system


def test_generate_state_ensemble_object_from_state_ensemble_name_object_name_is_physicality_required():
    # Case 1
    c_sys = generate_composite_system(mode="qubit", num=1, ids_esys=[0])
    actual = generate_state_ensemble_object_from_state_ensemble_name_object_name(
        state_ensemble_name="x0",
        object_name="state_ensemble",
        c_sys=c_sys,
        is_physicality_required=True,
    )

    # Assert
    expected = True
    for a in actual.states:
        assert a.is_physicality_required is expected

    # Case 2
    # Act
    actual = generate_state_ensemble_object_from_state_ensemble_name_object_name(
        state_ensemble_name="x0",
        object_name="state_ensemble",
        c_sys=c_sys,
        is_physicality_required=False,
    )

    # Assert
    expected = False
    for a in actual.states:
        assert a.is_physicality_required is expected

    # Case 3
    # Act
    actual = generate_state_ensemble_object_from_state_ensemble_name_object_name(
        state_ensemble_name="x0", object_name="state_ensemble", c_sys=c_sys
    )

    # Assert
    expected = True
    for a in actual.states:
        assert a.is_physicality_required is expected
