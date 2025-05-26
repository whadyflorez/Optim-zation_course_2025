import numpy as np
from scipy.optimize import minimize

# Función objetivo
def objective(x):
    x1, x2, x3 = x
    return (x1 - 1)**2 + (x2 - 2)**2 + (x3 - 3)**2 + x1 * x2 * x3

# Restricción de igualdad: x + y + z = 6
def eq_constraint(x):
    return x[0] + x[1] + x[2] - 6

# Restricción de desigualdad: x * y - z >= 1  →  x*y - z - 1 >= 0
def ineq_constraint(x):
    return x[0] * x[1] - x[2] - 1

# Definimos restricciones
constraints = [
    {'type': 'eq',   'fun': eq_constraint},
    {'type': 'ineq', 'fun': ineq_constraint}
]

# Bounds para cada variable: 0 ≤ x1 ≤ 5, 0 ≤ x2 ≤ 4, 1 ≤ x3 ≤ 3
bounds = [
    (0, 5),
    (0, 4),
    (1, 3)
]

# Punto inicial factible
x0 = np.array([2.0, 2.0, 2.0])

# Optimización con SLSQP
result = minimize(
    objective,
    x0,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints
)

# Resultados
print("Éxito:", result.success)
print("Mensaje  :", result.message)
print("Variables óptimas:", result.x)
print("Valor mínimo     :", result.fun)
print("Eq. restricción  (≈0):", eq_constraint(result.x))
print("Ineq. restricción(≥0):", ineq_constraint(result.x))
