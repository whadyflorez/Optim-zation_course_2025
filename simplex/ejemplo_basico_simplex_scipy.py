import numpy as np
from scipy.optimize import linprog

# ================================
# DEFINICIÓN DEL PROBLEMA
# ================================

# Función objetivo: Max F = x1 + 2x2 + x3
# Nota: linprog minimiza por defecto, por eso usamos los coeficientes negativos
c = [-1, -2, -1]  # Minimizar -F = -x1 - 2x2 - x3

# Restricciones de desigualdad:
#   2x1 +  x2 - x3 ≤  2
#  -2x1 +  x2 - 5x3 ≥ -6  → multiplicamos por -1 → 2x1 - x2 + 5x3 ≤ 6
#   4x1 +  x2 + x3 ≤ 6

A_ub = [
    [ 2,  1, -1],  # Restricción 1
    [ 2, -1,  5],  # Restricción 2 (reescrita)
    [ 4,  1,  1]   # Restricción 3
]

b_ub = [2, 6, 6]

# Restricciones de no negatividad: x1, x2, x3 ≥ 0
bounds = [(0, None), (0, None), (0, None)]

# ================================
# RESOLUCIÓN CON linprog (método simplex)
# ================================

result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='simplex')

# ================================
# SALIDA DETALLADA PARA ESTUDIANTES
# ================================

# Mostrar matrices del problema
print("=== MATRICES UTILIZADAS ===\n")
print("Vector de la función objetivo (c):")
print(np.array(c))

print("\nMatriz A_ub (coeficientes de las restricciones):")
print(np.array(A_ub))

print("\nVector b_ub (términos del lado derecho):")
print(np.array(b_ub))

print("\nRestricciones de no negatividad (bounds):")
print(bounds)

# Mostrar resultados de la optimización
print("\n=== RESULTADOS DE LA OPTIMIZACIÓN ===")
print("Mensaje del solver:", result.message)
print("¿La optimización fue exitosa?", result.success)
print("Número de iteraciones realizadas:", result.nit)
print("Valor óptimo de la función objetivo F:", -result.fun)  # Negamos porque maximizábamos
print("Solución óptima (valores de x):", result.x)
