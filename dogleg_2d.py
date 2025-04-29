
"""Dog‑leg / Trust‑region en 2‑D con visualización.

Para ejecutar: python dogleg_2d.py
"""
import numpy as np
import matplotlib.pyplot as plt

H = np.array([[3.0, 1.0],
              [1.0, 2.0]])
c = np.array([-2.0, -6.0])

def f(x):
    return 0.5 * x @ H @ x + c @ x

def grad(x):
    return H @ x + c

def hess(x):
    return H

def dogleg_2d(x0, delta0=1.0, delta_max=5.0,
              eta1=0.25, eta2=0.75,
              tol=1e-6, max_iter=20):
    x, delta = np.asarray(x0, float), float(delta0)
    path = []
    for _ in range(max_iter):
        g, B = grad(x), hess(x)
        if np.linalg.norm(g) < tol:
            path.append({'x': x.copy(), 'delta': delta}); break
        pB = -np.linalg.solve(B, g)
        gBg = g @ B @ g
        alpha = (g @ g) / gBg if gBg > 0 else delta / np.linalg.norm(g)
        pU = -alpha * g
        if np.linalg.norm(pB) <= delta:
            p = pB
        elif np.linalg.norm(pU) >= delta:
            p = delta * pU / np.linalg.norm(pU)
        else:
            diff = pB - pU
            a = diff @ diff
            b = 2 * (pU @ diff)
            cquad = pU @ pU - delta**2
            tau = (-b + np.sqrt(b**2 - 4*a*cquad)) / (2*a)
            p = pU + tau * diff
        rho = (f(x) - f(x+p)) / -(g @ p + 0.5*p @ B @ p)
        if rho < eta1:
            delta *= 0.5
        else:
            x = x + p
            if rho > eta2:
                delta = min(2*delta, delta_max)
        path.append({'x': x.copy(), 'delta': delta})
        if np.linalg.norm(g) < tol:
            break
    return x, path

if __name__ == "__main__":
    x_opt, steps = dogleg_2d([0,0])
    xs = [s['x'] for s in steps]
    deltas = [s['delta'] for s in steps]
    print("Optimo aproximado:", x_opt)
