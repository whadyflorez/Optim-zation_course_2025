import numpy as np
from scipy.linalg import solve
from tabulate import tabulate

# Sistema original Ax = b
A = np.array([
    [1, 2, 1, 0, 1],
    [2, 1, 3, 1, 0],
    [1, 1, 1, 2, 1]
], dtype=float)

b = np.array([8, 12, 10], dtype=float)

# Índices de las variables básicas (por ejemplo: x1, x2, x3)
basic_indices = [0, 1, 2]  # columnas de A correspondientes a x1, x2, x3
nonbasic_indices = [i for i in range(A.shape[1]) if i not in basic_indices]  # el resto: x4, x5

# Submatrices
A_B = A[:, basic_indices]  # matriz base (3x3)
A_N = A[:, nonbasic_indices]  # matriz de no básicas (3x2)

# Resolver A_B x_B = b para obtener la parte constante
const_vector = solve(A_B, b)

# Resolver A_B x_B = -A_N[:, j] para cada variable no básica (una columna por vez)
coeff_matrix = []
for j in range(len(nonbasic_indices)):
    rhs = -A_N[:, j]
    coeff_col = solve(A_B, rhs)
    coeff_matrix.append(coeff_col)

# Reorganizar coeficientes como matriz (cada columna corresponde a una variable no básica)
coeff_matrix = np.column_stack(coeff_matrix)

# Formatear resultados como ecuaciones
rows = []
for i, idx in enumerate(basic_indices):
    row = [f"x{idx+1} ="] + [f"{coeff_matrix[i,j]:.3f}" for j in range(len(nonbasic_indices))] + [f"{const_vector[i]:.3f}"]
    rows.append(row)

headers = ["Variable básica"] + [f"x{j+1}" for j in nonbasic_indices] + ["Constante"]

# Imprimir tabla
print(tabulate(rows, headers=headers))
