import time
from scipy.sparse import csr_matrix
from quara.objects.matrix_basis import MatrixBasis, SparseMatrixBasis
from quara.objects.composite_system_typical import generate_composite_system
from memory_profiler import profile

@profile
def main():
    c_sys = generate_composite_system(mode="qubit", num=4)
    _ = c_sys._calc_basis_basisconjugate_sparse()


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")
    print("処理が完了しました")
