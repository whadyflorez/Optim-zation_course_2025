# Ejemplo de problema lineal con múltiples soluciones óptimas

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# =============================================
# DEFINICIÓN DEL PROBLEMA
# =============================================
# Maximizar F = x1 + x2
# sujeto a:
#   x1 + x2 <= 4
#   x1 <= 2
#   x1, x2 >= 0

# Como linprog minimiza, usamos -F = -x1 - x2
c = [-1, -1]  # coeficientes de la función objetivo (minimizar -F)

# Restricciones en forma Ax <= b
A_ub = [
    [1, 1],  # x1 + x2 <= 4
    [1, 0]   # x1 <= 2
]
b_ub = [4, 2]

# Restricciones de no negatividad (por defecto)
bounds = [(0, None), (0, None)]

# =============================================
# RESOLVER PRIMERA VEZ
# =============================================
res1 = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

print("=== SOLUCIÓN ORIGINAL ===")
print("¿Éxito?:", res1.success)
print("F óptimo:", -res1.fun)
print("x óptimo:", res1.x)

# =============================================
# FIJAR x1 = 2 PARA OBTENER OTRA SOLUCIÓN
# =============================================
bounds_modified = [(2, 2), (0, None)]  # x1 fijo en 2
res2 = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_modified, method='highs')

print("\n=== SOLUCIÓN ALTERNA (x1 = 2) ===")
print("¿Éxito?:", res2.success)
print("F óptimo:", -res2.fun)
print("x óptimo:", res2.x)

# =============================================
# VISUALIZACIÓN GEOMÉTRICA
# =============================================
fig, ax = plt.subplots(figsize=(8, 6))

# Dibujar la región factible
x = np.linspace(0, 5, 200)
y1 = 4 - x       # x1 + x2 <= 4
x2_limit = 2     # x1 <= 2

plt.fill_between(x, 0, y1, where=(x <= x2_limit), color='lightgray', alpha=0.5, label='Región factible')

# Dibujar líneas de restricción
plt.plot(x, y1, 'r--', label='x1 + x2 = 4')
plt.axvline(x=2, color='b', linestyle='--', label='x1 = 2')

# Dibujar soluciones
if res1.success:
    plt.plot(res1.x[0], res1.x[1], 'ko', label='Solución óptima 1')
if res2.success:
    plt.plot(res2.x[0], res2.x[1], 'mo', label='Solución óptima 2')

# Dibujar combinaciones convexas entre las soluciones
if res1.success and res2.success:
    x1 = res1.x
    x2 = res2.x
    for lam in np.linspace(0, 1, 5):
        x_comb = lam * x1 + (1 - lam) * x2
        plt.plot(x_comb[0], x_comb[1], 'gx')
        plt.text(x_comb[0]+0.05, x_comb[1], f"λ={lam:.2f}", fontsize=8)

# Configuración del gráfico
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Familia de soluciones óptimas en un problema convexo")
plt.xlim(0, 5)
plt.ylim(0, 5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
