"""Gradient descent with *exact* 1‑D line‑search (golden section).

Run:
    python gd_exact_ls_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (1 - x[0])**2 + x[1]**4 


def grad_f(x):
    dfdx = -2*(1-x[0])
    dfdy = 4*x[1]**3
    return np.array([dfdx, dfdy])


# ---------- golden‑section search ----------
def golden_section_search(phi, a, b, tol=1e-5, max_iter=100):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    fc, fd = phi(c), phi(d)

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, fd = d, fc
            d = c
            c = b - (b - a) / gr
            fc = phi(c)
        else:
            a, fc = c, fd
            c = d
            d = a + (b - a) / gr
            fd = phi(d)
    return (a + b) / 2


def line_search_min_alpha(x, p, alpha_max=1.0):
    phi = lambda a: f(x + a * p)
    return golden_section_search(phi, 0.0, alpha_max)


def gd_exact_linesearch(x0, max_iter=3, alpha_max=1.0):
    path = [x0.copy()]
    alphas = []
    x = x0.copy()
    for _ in range(max_iter):
        g = grad_f(x)
        p = -g
        alpha = line_search_min_alpha(x, p, alpha_max)
        x = x + alpha * p
        path.append(x.copy())
        alphas.append(alpha)
    return path, alphas


def main():
    x0 = np.array([-1.2, 1.0])
    path, alphas = gd_exact_linesearch(x0, max_iter=10)

    # Table
    print("Iter |      x          y        f(x,y)        α")
    print("--------------------------------------------------")
    for k, x in enumerate(path):
        fx = f(x)
        if k == 0:
            print(f"{k:>4} | {x[0]: .6f} {x[1]: .6f} {fx: .6f}      ----")
        else:
            print(f"{k:>4} | {x[0]: .6f} {x[1]: .6f} {fx: .6f}  {alphas[k-1]: .6f}")

    # Plot
    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z=(1 - X)**2 + Y**4

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=40, linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    path_arr = np.array(path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], marker="o", linewidth=1.5, markersize=5)
    plt.title("Gradient Descent with 1‑D Exact Line‑Search (3 steps)")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
