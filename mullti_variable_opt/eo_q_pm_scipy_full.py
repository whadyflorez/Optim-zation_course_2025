"""
eo_q_pm_scipy_full.py
=====================
Optimiza simultáneamente el tamaño de lote de producción (EOQ) y el intervalo
de mantenimiento preventivo (PM) para minimizar el costo anual total.

Requiere:
    numpy
    scipy
    matplotlib  (solo para la gráfica, opcional)

Ejecuta:
    python eo_q_pm_scipy_full.py
"""

import numpy as np
from scipy.optimize import minimize

# ------------- parámetros del problema -------------
D = 10_000        # demanda anual (u/año)
S = 200.0         # costo de setup por lote ($)
h = 1.0           # almacenamiento ($/u·año)
C_PM = 500.0      # mantenimiento preventivo ($)
C_FAIL = 10_000.0 # costo de falla ($)
lam = 0.02        # tasa de fallas (1/año)

# ------------- costo total y gradiente -------------
def total_cost(x):
    Q, T = x
    if np.any(x <= 0):
        return np.inf
    return (D / Q) * S + h * Q / 2 + C_PM / T + C_FAIL * (1 - np.exp(-lam * T))

def grad_total_cost(x):
    Q, T = x
    dC_dQ = -D * S / Q**2 + h / 2
    dC_dT = -C_PM / T**2 + C_FAIL * lam * np.exp(-lam * T)
    return np.array([dC_dQ, dC_dT])

# ------------- optimización (L‑BFGS‑B) --------------
x0 = np.array([400.0, 20.0])          # punto inicial
bounds = [(1e-6, None), (1e-6, None)] # Q > 0, T > 0

path = []                              # para guardar la trayectoria
callback = lambda xk: path.append(xk.copy())

res = minimize(
    total_cost,
    x0,
    method="L-BFGS-B",
    jac=grad_total_cost,
    bounds=bounds,
    callback=callback,
    options={"ftol": 1e-12, "maxiter": 200},
)

opt_Q, opt_T = res.x
opt_cost = res.fun

print("\n=== Política óptima EOQ + PM ===")
print(f"Q* (lote) : {opt_Q:8.2f} unidades")
print(f"T* (PM)   : {opt_T:8.2f} años")
print(f"Costo min : ${opt_cost:,.2f}")

# ------------- gráfica de contornos -----------------
try:
    import matplotlib.pyplot as plt

    # superficie de costo para visualizar
    Q_vals = np.linspace(50, 800, 600)
    T_vals = np.linspace(5, 60, 600)
    QQ, TT = np.meshgrid(Q_vals, T_vals)
    Z = (D / QQ) * S + h * QQ / 2 + C_PM / TT + C_FAIL * (1 - np.exp(-lam * TT))

    plt.figure(figsize=(7, 5))
    cs = plt.contour(QQ, TT, Z, levels=40, linewidths=0.5, cmap="viridis")
    plt.clabel(cs, inline=True, fontsize=6, fmt="%.0f")

    path_arr = np.vstack(path)
    plt.plot(
        path_arr[:, 0],
        path_arr[:, 1],
        "-o",
        color="crimson",
        linewidth=1.5,
        markersize=4,
        label="Trayectoria L‑BFGS‑B",
    )
    plt.scatter(opt_Q, opt_T, marker="*", s=150, color="gold", label="Óptimo")
    plt.xlabel("Tamaño de lote  Q  (unidades)")
    plt.ylabel("Intervalo PM  T  (años)")
    plt.title("Superficie de costo C(Q,T) con trayectoria de optimización")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

except ImportError:
    # matplotlib no disponible; solo resultados numéricos
    pass
