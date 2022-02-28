from itertools import product
from typing import List, Union
import numpy as np

from quara.objects.composite_system import CompositeSystem
from quara.objects.gate import convert_hs
from quara.objects.mprocess import MProcess
from quara.utils.matrix_util import calc_mat_from_vector_adjoint, truncate_hs


def get_mprocess_object_names() -> List[str]:
    """returns the list of valid mprocess-related object names.

    Returns
    -------
    List[str]
        the list of valid mprocess-related object names.
    """
    names = [
        "set_pure_state_vectors",
        "set_kraus_matrices",
        "hss",
        "mprocess",
    ]
    return names


def get_mprocess_names_type1_set_pure_state_vectors() -> List[str]:
    """returns the list of valid MProcess names of `set pure vectors` of type1.

    Returns
    -------
    List[str]
        the list of valid MProcess names of `set pure vectors` of type1.
    """
    names = [
        "x-type1",
        "y-type1",
        "z-type1",
        "bell-type1",
        "z3-type1",
        "z2-type1",
    ]
    return names


def get_mprocess_names_type1_set_kraus_matrices() -> List[str]:
    """returns the list of valid MProcess names of `set Kraus matrices` of type1.

    Returns
    -------
    List[str]
        the list of valid MProcess names of `set Kraus matrices` of type1.
    """
    names = [
        "xxparity-type1",
        "zzparity-type1",
    ]
    return names


def get_mprocess_names_type1() -> List[str]:
    """returns the list of valid MProcess names of type1.

    Returns
    -------
    List[str]
        the list of valid MProcess names of type1.
    """
    names = (
        get_mprocess_names_type1_set_pure_state_vectors()
        + get_mprocess_names_type1_set_kraus_matrices()
    )
    return names


def get_mprocess_names_type2() -> List[str]:
    """returns the list of valid MProcess names of type2.

    Returns
    -------
    List[str]
        the list of valid MProcess names of type2.
    """
    names = [
        "x-type2",
        "y-type2",
        "z-type2",
        "z3-type2",
        "z2-type2",
    ]
    return names


def generate_mprocess_set_pure_state_vectors_from_name(
    mprocess_name: str,
) -> List[List[np.ndarray]]:
    """returns the set of pure state vectors of MProcess specified by name.

    Parameters
    ----------
    mprocess_name : str
        name of the MProcess.

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess

    Raises
    ------
    ValueError
        mprocess_name is invalid.
    """
    # split and get each pure state vectors
    single_mprocess_names = mprocess_name.split("_")
    set_pure_state_vectors_list = [
        _generate_mprocess_set_pure_state_vectors_from_single_name(single_mprocess_name)
        for single_mprocess_name in single_mprocess_names
    ]

    # tensor product
    temp = set_pure_state_vectors_list[0]
    for pure_state_vectors in set_pure_state_vectors_list[1:]:
        temp = [np.kron(vec1, vec2) for vec1, vec2 in product(temp, pure_state_vectors)]

    return temp


def _generate_mprocess_set_pure_state_vectors_from_single_name(
    mprocess_name: str,
) -> List[np.ndarray]:
    if mprocess_name not in get_mprocess_names_type1_set_pure_state_vectors():
        raise ValueError(f"mprocess_name is not type 1. mprocess_name={mprocess_name}")

    method_name = (
        "get_mprocess_" + mprocess_name.replace("-", "") + "_set_pure_state_vectors"
    )
    method = eval(method_name)
    set_pure_state_vectors = method()
    return set_pure_state_vectors


def generate_mprocess_set_kraus_matrices_from_name(
    mprocess_name: str,
) -> List[List[np.ndarray]]:
    """returns the set of Kraus matrices of MProcess specified by name.

    Parameters
    ----------
    mprocess_name : str
        name of the MProcess.

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices

    Raises
    ------
    ValueError
        mprocess_name is invalid.
    """
    # split and get each pure state vectors
    single_mprocess_names = mprocess_name.split("_")
    set_kraus_matrices_list = [
        _generate_mprocess_set_kraus_matrices_from_name_from_single_name(
            single_mprocess_name
        )
        for single_mprocess_name in single_mprocess_names
    ]

    # tensor product
    temp = set_kraus_matrices_list[0]
    for set_kraus_matrices in set_kraus_matrices_list[1:]:
        temp = [np.kron(mat1, mat2) for mat1, mat2 in product(temp, set_kraus_matrices)]

    return temp


def _generate_mprocess_set_kraus_matrices_from_name_from_single_name(
    mprocess_name: str,
) -> List[List[np.ndarray]]:
    set_kraus_matrices = []
    if mprocess_name in get_mprocess_names_type1_set_pure_state_vectors():
        set_pure_state_vectors = generate_mprocess_set_pure_state_vectors_from_name(
            mprocess_name
        )

        for pure_state_vectors in set_pure_state_vectors:
            matrices = [
                calc_mat_from_vector_adjoint(pure_state_vector)
                for pure_state_vector in pure_state_vectors
            ]
            set_kraus_matrices.append(matrices)
    elif (
        mprocess_name in get_mprocess_names_type1_set_kraus_matrices()
        or mprocess_name in get_mprocess_names_type2()
    ):
        method_name = (
            "get_mprocess_" + mprocess_name.replace("-", "") + "_set_kraus_matrices"
        )
        method = eval(method_name)
        set_kraus_matrices = method()
    else:
        raise ValueError(
            f"mprocess_name is out of range. mprocess_name={mprocess_name}"
        )

    return set_kraus_matrices


def generate_mprocess_hss_from_name(
    mprocess_name: str, c_sys: CompositeSystem
) -> List[np.ndarray]:
    """returns the list of HS matrices of MProcess specified by name.

    Parameters
    ----------
    mprocess_name : str
        name of the MProcess.
    c_sys : CompositeSystem
        CompositeSystem of MProcess.

    Returns
    -------
    List[np.ndarray]
        the list of HS matrices

    Raises
    ------
    ValueError
        mprocess_name is invalid.
    """
    # check mprocess name
    single_mprocess_names = mprocess_name.split("_")
    mprocess_name_list = get_mprocess_names_type1() + get_mprocess_names_type2()
    for single_mprocess_name in single_mprocess_names:
        if single_mprocess_name not in mprocess_name_list:
            raise ValueError(
                f"mprocess_name is out of range. mprocess_name={single_mprocess_name}"
            )

    # generate hss
    size = c_sys.dim ** 2
    hss_cb = []
    set_kraus_matrices = generate_mprocess_set_kraus_matrices_from_name(mprocess_name)
    for kraus_matrices in set_kraus_matrices:
        tmp_hs = np.zeros((size, size), dtype=np.complex128)
        for kraus_matrix in kraus_matrices:
            tmp_hs += np.kron(kraus_matrix, kraus_matrix.conjugate())
        hss_cb.append(tmp_hs)
    hss = [
        truncate_hs(convert_hs(hs_cb, c_sys.comp_basis(), c_sys.basis()))
        for hs_cb in hss_cb
    ]

    return hss


def generate_mprocess_from_name(
    c_sys: CompositeSystem, mprocess_name: str, is_physicality_required: bool = True
) -> MProcess:
    """returns MProcess object specified by name.

    Parameters
    ----------
    c_sys : CompositeSystem
        CompositeSystem of MProcess.
    mprocess_name : str
        name of the MProcess.
    is_physicality_required: bool = True
        whether the generated object is physicality required, by default True

    Returns
    -------
    MProcess
        MProcess object.
    """

    # check mprocess name
    single_mprocess_names = mprocess_name.split("_")
    mprocess_name_list = get_mprocess_names_type1() + get_mprocess_names_type2()
    for single_mprocess_name in single_mprocess_names:
        if single_mprocess_name not in mprocess_name_list:
            raise ValueError(
                f"mprocess_name is out of range. mprocess_name={single_mprocess_name}"
            )

    # generate mprocess
    hss = generate_mprocess_hss_from_name(mprocess_name, c_sys)
    mprocess = MProcess(
        hss=hss, c_sys=c_sys, is_physicality_required=is_physicality_required
    )
    return mprocess


def generate_mprocess_object_from_mprocess_name_object_name(
    mprocess_name: str,
    object_name: str,
    c_sys: CompositeSystem = None,
    is_physicality_required: bool = True,
) -> Union[List[List[np.ndarray]], List[np.ndarray], MProcess]:
    """returns a mprocess-related object.

    Parameters
    ----------
    mprocess_name : str
        name of the MProcess.
    object_name : str
        Valid object_name. It is given by :func:`~quara.objects.mprocess_typical.get_mprocess_object_names`.
    c_sys : CompositeSystem, optional
        To be given for object_name='hss' or 'mprocess', by default None.
    is_physicality_required: bool = True
        whether the generated object is physicality required, by default True

    Returns
    -------
    Union[List[List[np.ndarray]], List[np.ndarray], MProcess]
        .. line-block::

            List[List[np.ndarray]]
                the set of pure state vectors of MProcess for object_name = 'set_pure_state_vectors'
                    Complex vectors
                the set of Kraus matrices of MProcess for object_name = 'set_kraus_matrices'
                    Complex matrices
            List[np.ndarray]
                the list of HS matrices of MProcess for object_name = 'hss'
                    Complex matrices
            MProcess
                MProcess class for object_name = 'mprocess'

    Raises
    ------
    ValueError
        mprocess_name is invalid.
    """
    if object_name not in get_mprocess_object_names():
        raise ValueError("object_name is out of range.")
    elif object_name == "set_pure_state_vectors":
        return generate_mprocess_set_pure_state_vectors_from_name(mprocess_name)
    elif object_name == "set_kraus_matrices":
        return generate_mprocess_set_kraus_matrices_from_name(mprocess_name)
    elif object_name == "hss":
        return generate_mprocess_hss_from_name(mprocess_name, c_sys)
    elif object_name == "mprocess":
        return generate_mprocess_from_name(
            c_sys, mprocess_name, is_physicality_required=is_physicality_required
        )


def get_mprocess_xtype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    """return the set of pure state vectors of MProcess for `x-type1`.

    :math:`|\\psi_{0,0}\\rangle = |+\\rangle`

    :math:`|\\psi_{1,0}\\rangle = |-\\rangle`

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess for `x-type1`.
    """
    set_pure_state_vectors = [
        [
            (1 / np.sqrt(2)) * np.array([1, 1], dtype=np.complex128),
        ],
        [
            (1 / np.sqrt(2)) * np.array([1, -1], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_ytype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    """return the set of pure state vectors of MProcess for `y-type1`.

    :math:`|\\psi_{0,0}\\rangle = |+i\\rangle`

    :math:`|\\psi_{1,0}\\rangle = |-i\\rangle`

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess for `y-type1`.
    """
    set_pure_state_vectors = [
        [
            (1 / np.sqrt(2)) * np.array([1, 1j], dtype=np.complex128),
        ],
        [
            (1 / np.sqrt(2)) * np.array([1, -1j], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_ztype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    """return the set of pure state vectors of MProcess for `z-type1`.

    :math:`|\\psi_{0,0}\\rangle = |0\\rangle`

    :math:`|\\psi_{1,0}\\rangle = |1\\rangle`

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess for `z-type1`.
    """
    set_pure_state_vectors = [
        [
            np.array([1, 0], dtype=np.complex128),
        ],
        [
            np.array([0, 1], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_z3type1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    """return the set of pure state vectors of MProcess for `z3-type1`.

    :math:`|\\psi_{0,0}\\rangle = |0\\rangle`

    :math:`|\\psi_{1,0}\\rangle = |1\\rangle`

    :math:`|\\psi_{2,0}\\rangle = |2\\rangle`

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess for `z3-type1`.
    """
    set_pure_state_vectors = [
        [
            np.array([1, 0, 0], dtype=np.complex128),
        ],
        [
            np.array([0, 1, 0], dtype=np.complex128),
        ],
        [
            np.array([0, 0, 1], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_z2type1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    """return the set of pure state vectors of MProcess for `z2-type1`.

    :math:`|\\psi_{0,0}\\rangle = |0\\rangle`

    :math:`|\\psi_{1,0}\\rangle = |1\\rangle`

    :math:`|\\psi_{1,1}\\rangle = |2\\rangle`

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess for `z2-type1`.
    """
    set_pure_state_vectors = [
        [
            np.array([1, 0, 0], dtype=np.complex128),
        ],
        [
            np.array([0, 1, 0], dtype=np.complex128),
            np.array([0, 0, 1], dtype=np.complex128),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_belltype1_set_pure_state_vectors() -> List[List[np.ndarray]]:
    """return the set of pure state vectors of MProcess for `bell-type1`.

    :math:`|\\psi_{0,0}\\rangle = |\\Psi^+\\rangle = |0\\rangle|1\\rangle + |1\\rangle|0\\rangle`

    :math:`|\\psi_{1,0}\\rangle = |\\Psi^-\\rangle = |0\\rangle|1\\rangle - |1\\rangle|0\\rangle`

    :math:`|\\psi_{2,0}\\rangle = |\\Phi^+\\rangle = |0\\rangle|0\\rangle + |1\\rangle|1\\rangle`

    :math:`|\\psi_{3,0}\\rangle = |\\Phi^-\\rangle = |0\\rangle|0\\rangle - |1\\rangle|1\\rangle`

    Returns
    -------
    List[List[np.ndarray]]
        the set of pure state vectors of MProcess for `bell-type1`.
    """
    set_pure_state_vectors = [
        [
            np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2),
        ],
        [
            np.array([1, 0, 0, -1], dtype=np.complex128) / np.sqrt(2),
        ],
        [
            np.array([0, 1, 1, 0], dtype=np.complex128) / np.sqrt(2),
        ],
        [
            np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2),
        ],
    ]
    return set_pure_state_vectors


def get_mprocess_xtype2_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `x-type2`.

    :math:`K_{0,0} = |+\\rangle\\langle +|`

    :math:`K_{1,0} = |+\\rangle\\langle -|`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `x-type2`.
    """
    set_kraus_matrices = [
        [
            np.array([[1, 1], [1, 1]], dtype=np.complex128) / 2,
        ],
        [
            np.array([[1, -1], [1, -1]], dtype=np.complex128) / 2,
        ],
    ]
    return set_kraus_matrices


def get_mprocess_ytype2_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `y-type2`.

    :math:`K_{0,0} = |+i\\rangle\\langle +i|`

    :math:`K_{1,0} = |+i\\rangle\\langle -i|`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `y-type2`.
    """
    set_kraus_matrices = [
        [
            np.array([[1, -1j], [1j, 1]], dtype=np.complex128) / 2,
        ],
        [
            np.array([[1, 1j], [1j, -1]], dtype=np.complex128) / 2,
        ],
    ]
    return set_kraus_matrices


def get_mprocess_ztype2_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `z-type2`.

    :math:`K_{0,0} = |0\\rangle\\langle 0|`

    :math:`K_{1,0} = |0\\rangle\\langle 1|`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `z-type2`.
    """
    set_kraus_matrices = [
        [
            np.array([[1, 0], [0, 0]], dtype=np.complex128),
        ],
        [
            np.array([[0, 1], [0, 0]], dtype=np.complex128),
        ],
    ]
    return set_kraus_matrices


def get_mprocess_xxparitytype1_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `xx-parity`.

    :math:`K_{0,0} = \\frac{1}{2}\\begin{pmatrix} 1 & 0 & 0 & 1 \\\\ 0 & 1 & 1 & 0 \\\\ 0 & 1 & 1 & 0 \\\\ 1 & 0 & 0 & 1 \\end{pmatrix}`

    :math:`K_{1,0} = \\frac{1}{2}\\begin{pmatrix} 1 & 0 & 0 & -1 \\\\ 0 & 1 & -1 & 0 \\\\ 0 & -1 & 1 & 0 \\\\ -1 & 0 & 0 & 1 \\end{pmatrix}`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `xx-parity`.
    """
    k0 = 0.5 * np.array(
        [[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.complex128
    )
    k1 = 0.5 * np.array(
        [[1, 0, 0, -1], [0, 1, -1, 0], [0, -1, 1, 0], [-1, 0, 0, 1]],
        dtype=np.complex128,
    )
    return [[k0], [k1]]


def get_mprocess_zzparitytype1_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `zz-parity`.

    :math:`K_{0,0} = \\begin{pmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 0 \\\\ 0 & 0 & 0 & 1 \\end{pmatrix}`

    :math:`K_{1,0} = \\begin{pmatrix} 0 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\\\ 0 & 0 & 1 & 0 \\\\ 0 & 0 & 0 & 0 \\end{pmatrix}`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `zz-parity`.
    """
    k0 = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=np.complex128
    )
    k1 = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=np.complex128
    )
    return [[k0], [k1]]


def get_mprocess_z3type2_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `z3-type2`.

    :math:`K_{0,0} = |0\\rangle\\langle 0|`

    :math:`K_{1,0} = |0\\rangle\\langle 1|`

    :math:`K_{2,0} = |0\\rangle\\langle 2|`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `z3-type2`.
    """
    set_kraus_matrices = [
        [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
        ],
        [
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
        ],
        [
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
        ],
    ]
    return set_kraus_matrices


def get_mprocess_z2type2_set_kraus_matrices() -> List[List[np.ndarray]]:
    """return the set of Kraus matrices of MProcess for `z2-type2`.

    :math:`K_{0,0} = |0\\rangle\\langle 0|`

    :math:`K_{1,0} = |0\\rangle\\langle 1|`

    :math:`K_{1,1} = |0\\rangle\\langle 2|`

    Returns
    -------
    List[List[np.ndarray]]
        the set of Kraus matrices of MProcess for `z2-type2`.
    """
    set_kraus_matrices = [
        [
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
        ],
        [
            np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
            np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.complex128),
        ],
    ]
    return set_kraus_matrices
