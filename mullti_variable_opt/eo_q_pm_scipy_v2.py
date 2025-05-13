
# eo_q_pm_scipy_v2.py  –  Improved example with trajectory and L‑BFGS‑B
import numpy as np
from scipy.optimize import minimize

D, S, h = 10_000, 200.0, 1.0
C_PM, C_FAIL, lam = 500.0, 10_000.0, 0.02

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

x0 = np.array([100.0, 2.0])

path = []
res = minimize(
    total_cost, x0,
    method="Newton-CG",
    jac=grad_total_cost,
    callback=lambda xk: path.append(xk.copy()),
    options={"xtol": 1e-6, "maxiter": 200},
)

print("Optimal Q, T :", res.x)
print("Minimum cost :", res.fun)
