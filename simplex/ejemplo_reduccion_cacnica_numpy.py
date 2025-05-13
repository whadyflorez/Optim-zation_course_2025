import numpy as np

# Sistema: Ax = b
A = np.array([
    [1, 2, 1, 0, 1],
    [2, 1, 3, 1, 0],
    [1, 1, 1, 2, 1]
])
b = np.array([8, 12, 10])

# Definir qué columnas son básicas (ej: x1, x2, x3 → columnas 0,1,2)
basic_indices = [0, 1, 2]
nonbasic_indices = [i for i in range(A.shape[1]) if i not in basic_indices]

# Extraer submatrices
A_B = A[:, basic_indices]  # matriz base
A_N = A[:, nonbasic_indices]  # matriz de no básicas

# Invertir A_B
A_B_inv = np.linalg.inv(A_B)

# Forma canónica: x_B = A_B⁻¹ (b - A_N x_N)
# => x_B = constante + coeficientes * x_N
coeff_matrix = -A_B_inv @ A_N
const_vector = A_B_inv @ b

# Mostrar resultados
from tabulate import tabulate

rows = []
for i, idx in enumerate(basic_indices):
    row = [f"x{idx+1} ="] + [f"{coeff_matrix[i,j]:.3f}" for j in range(len(nonbasic_indices))] + [f"{const_vector[i]:.3f}"]
    rows.append(row)

headers = ["Variable básica"] + [f"x{j+1}" for j in nonbasic_indices] + ["Constante"]

print(tabulate(rows, headers=headers))

