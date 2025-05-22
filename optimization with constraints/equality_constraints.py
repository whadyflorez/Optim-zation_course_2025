import numpy as np
from scipy.optimize import minimize

# Definimos la función objetivo
def objective(x):
    x1, x2, x3 = x
    return (x1 - 1)**2 + (x2 - 2)**2 + (x3 - 3)**2 + x1 * x2 * x3

# Definimos las restricciones de igualdad
def constraint1(x):
    return x[0] + x[1] + x[2] - 6  # x + y + z = 6

def constraint2(x):
    return x[0]**2 + x[1]**2 + x[2]**2 - 14  # x^2 + y^2 + z^2 = 14

constraints = [
    {'type': 'eq', 'fun': constraint1},
    {'type': 'eq', 'fun': constraint2}
]

# Punto inicial que satisface las restricciones
x0 = np.array([2.0, 1.0, 3.0])

# Llamada a scipy.optimize.minimize con SLSQP
result = minimize(
    objective,
    x0,
    method='SLSQP',
    constraints=constraints
)

# Mostrar resultados
print("Éxito:", result.success)
print("Mensaje  :", result.message)
print("Variables óptimas:", result.x)
print("Valor mínimo     :", result.fun)
print("Restricción 1 (≈0):", constraint1(result.x))
print("Restricción 2 (≈0):", constraint2(result.x))
