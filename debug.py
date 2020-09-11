import pickle
import numpy as np

path = "tutorials/qst_data/z0_linear_20200909_114932.pkl"

with open(path, "rb") as f:
    param_linear_est_linear = pickle.load(f)

i = 184
print(i)
for j in range(4):
    print(f"vec[{j}]: {param_linear_est_linear[i][0].vec[j]}")
    print(f"is_trace_one: {param_linear_est_linear[i][0].is_trace_one()}")
density_matrix = param_linear_est_linear[i][0].to_density_matrix()

# print(f"Result of to_density_matirx")
# print(f"(0, 0) = {density_matrix[0][0]}")
# print(f"(0, 1) = {density_matrix[0][1]}")
# print(f"(1, 0) = {density_matrix[1][0]}")
# print(f"(1, 1) = {density_matrix[1][1]}")

# eig = np.linalg.eigvals(density_matrix)

# print("Eigenvalues:")
# print(f"eigenvalues[0] = {eig[0]}")
# print(f"eigenvalues[1] = {eig[1]}")

print("completed")
