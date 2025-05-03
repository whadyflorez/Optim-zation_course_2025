
"""Newton's method with exact 1‑D line‑search on the Rosenbrock function.

If SciPy is installed, the step length is determined by
scipy.optimize.minimize_scalar (bounded); otherwise a simple
golden‑section search is used.

Run:
    python newton_ls_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt

# Try SciPy for 1‑D minimisation
try:
    from scipy.optimize import minimize_scalar
    have_scipy = True
except ImportError:
    have_scipy = False


def f(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2


def grad_f(x):
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    dfdy = 200 * (x[1] - x[0]**2)
    return np.array([dfdx, dfdy])


def hess_f(x):
    x0, y0 = x
    h11 = 2 - 400 * y0 + 1200 * x0**2
    h12 = -400 * x0
    h22 = 200
    return np.array([[h11, h12],
                     [h12, h22]])


def golden_section_search(phi, a, b, tol=1e-6, max_iter=100):
    gr = (np.sqrt(5) + 1)/2
    c = b - (b - a)/gr
    d = a + (b - a)/gr
    fc, fd = phi(c), phi(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - (b - a)/gr
            fc = phi(c)
        else:
            a, fc = c, fd
            c = d
            d = a + (b - a)/gr
            fd = phi(d)
    return (a + b)/2


def line_search_alpha(x, p, alpha_max=1.0):
    phi = lambda a: f(x + a * p)
    if have_scipy:
        res = minimize_scalar(phi, bounds=(0.0, alpha_max), method='bounded')
        return res.x
    return golden_section_search(phi, 0.0, alpha_max)


def newton_linesearch(x0, tol=1e-8, max_iter=20):
    path = [x0.copy()]
    alphas, grad_norms = [], []
    x = x0.copy()
    for _ in range(max_iter):
        g = grad_f(x)
        grad_norms.append(np.linalg.norm(g))
        if grad_norms[-1] < tol:
            break
        H = hess_f(x)
        try:
            p = -np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            p = -g
        if np.dot(g, p) >= 0:
            p = -g
        alpha = line_search_alpha(x, p)
        x = x + alpha * p
        path.append(x.copy())
        alphas.append(alpha)
    return np.array(path), alphas, grad_norms


def main():
    x0 = np.array([-1.2, 1.0])
    path, alphas, grad_norms = newton_linesearch(x0)
    print(f"Newton converged in {len(path)-1} iterations")

    print("Iter |      x          y        f(x,y)        |∇f|       α")
    print("------------------------------------------------------------------")
    for k, x in enumerate(path):
        fx = f(x)
        if k == 0:
            print(f"{k:>4} | {x[0]: .6f} {x[1]: .6f} {fx: .6e} {grad_norms[0]: .3e}    ----")
        else:
            print(f"{k:>4} | {x[0]: .6f} {x[1]: .6f} {fx: .6e} {grad_norms[k-1]: .3e} {alphas[k-1]: .3e}")

    # Plot trajectory
    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X)**2 + 100 * (Y - X**2)**2

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 40), linewidths=0.5, norm="log")
    plt.clabel(cs, inline=True, fontsize=6, fmt="%.1e")
    plt.plot(path[:, 0], path[:, 1], marker="o", linewidth=1.5, markersize=5)
    plt.title("Newton Method with 1‑D Exact Line‑Search")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
