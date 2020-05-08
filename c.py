import itertools

import numpy as np
import numpy.testing as npt
import pytest

from quara.objects import matrix_basis
from quara.objects.composite_system import CompositeSystem
from quara.objects.elemental_system import ElementalSystem
from quara.objects.gate import (
    Gate,
    get_cnot,
    get_cz,
    get_h,
    get_i,
    get_root_x,
    get_root_y,
    get_s,
    get_sdg,
    get_swap,
    get_t,
    get_x,
    get_y,
    get_z,
)
from quara.objects.operators import (
    _composite,
    _tensor_product,
    _to_list,
    composite,
    tensor_product,
)
from quara.objects.povm import (
    Povm,
    get_x_measurement,
    get_xx_measurement,
    get_xy_measurement,
    get_xz_measurement,
    get_y_measurement,
    get_yx_measurement,
    get_yy_measurement,
    get_yz_measurement,
    get_z_measurement,
    get_zx_measurement,
    get_zy_measurement,
    get_zz_measurement,
)
from quara.objects.state import (
    State,
    get_x0_1q,
    get_x1_1q,
    get_y0_1q,
    get_z0_1q,
    get_z1_1q,
)

basis1 = matrix_basis.get_comp_basis()
e_sys1 = ElementalSystem(1, basis1)
c_sys1 = CompositeSystem([e_sys1])
vecs1 = [
    np.array([2, 3, 5, 7], dtype=np.float64),
    np.array([11, 13, 17, 19], dtype=np.float64),
]
povm1 = Povm(c_sys1, vecs1, is_physical=False)

basis2 = matrix_basis.get_comp_basis()
e_sys2 = ElementalSystem(2, basis2)
c_sys2 = CompositeSystem([e_sys2])
vecs2 = [
    np.array([23, 29, 31, 37], dtype=np.float64),
    np.array([41, 43, 47, 53], dtype=np.float64),
]
povm2 = Povm(c_sys2, vecs2, is_physical=False)

povm12 = tensor_product(povm1, povm2)

print("complete")
