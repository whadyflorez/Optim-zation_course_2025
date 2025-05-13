from scipy.optimize import linprog

# ============================================
# PROBLEMA UNBOUNDED (NO ACOTADO)
# ============================================

# Queremos: Max F = x1 + x2
# Pero linprog minimiza, así que usamos: -F = -x1 - x2
c = [-1, -1]  # coeficientes de la función objetivo

# Restricción: x1 - x2 ≥ 1  → se escribe como -x1 + x2 ≤ -1
A_ub = [[-1, 1]]
b_ub = [-1]

# Restricciones de no negatividad:
# x1 ≥ 0, x2 sin cota superior → esto genera un problema no acotado
bounds = [(0, None), (None, None)]

# ============================================
# RESOLVER CON SCIPY.LINPROG (método 'highs')
# ============================================

result = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

# ============================================
# DIAGNÓSTICO DETALLADO
# ============================================

print("===== DIAGNÓSTICO DE PROBLEMA UNBOUNDED =====\n")
print("¿La optimización fue exitosa?:", result.success)
print("Código de estado             :", result.status)
print("Mensaje del solver           :", result.message)

if result.success:
    print("\nValor óptimo de F:", -result.fun)
    print("Solución óptima encontrada (x):", result.x)
else:
    print("\n→ No se encontró solución porque el problema es no acotado.")
