"""
Gradient‑descent with Armijo line‑search — classroom demo
========================================================
Minimises the 2‑D Rosenbrock function and plots the first
three iterations, illustrating how the back‑tracking
algorithm chooses the step length α_k.

Run:
    python gd_line_search_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x: np.ndarray) -> float:
    """Rosenbrock function."""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def grad_f(x: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock function."""
    dfdx = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dfdy = 200 * (x[1] - x[0] ** 2)
    return np.array([dfdx, dfdy])


def gradient_descent_line_search(
    x0: np.ndarray,
    max_iter: int = 100,
    alpha0: float = 0.01,
    rho: float = 0.5,
    c: float = 1e-8,
):
    """
    Gradient descent with Armijo back‑tracking line‑search.

    Parameters
    ----------
    x0 : ndarray
        Initial point.
    max_iter : int, optional
        Number of iterations to perform. Default 3.
    alpha0 : float, optional
        Initial trial step length. Default 1.0.
    rho : float, optional
        Contraction factor (0 < rho < 1). Default 0.5.
    c : float, optional
        Armijo parameter (0 < c < 1). Default 1e‑4.

    Returns
    -------
    path : list[ndarray]
        List of points visited (including x0).
    alphas : list[float]
        Accepted step lengths per iteration.
    """
    x = x0.copy()
    path = [x.copy()]
    alphas = []

    for _ in range(max_iter):
        g = grad_f(x)
        p = -g                       # steepest‑descent direction
        alpha = alpha0
        fx = f(x)

        # Armijo test
        while f(x + alpha * p) > fx + c * alpha * np.dot(g, p):
            alpha *= rho

        x = x + alpha * p
        path.append(x.copy())
        alphas.append(alpha)

    return path, alphas


def main():
    # ------------------------------------------------------------------
    # Run a 3‑step demo
    x0 = np.array([-1.2, 1.0])
    path, alphas = gradient_descent_line_search(x0, max_iter=3)

    # Print a small table
    print("Iter |      x         y        f(x,y)        α")
    print("--------------------------------------------------")
    for k, x in enumerate(path):
        fx = f(x)
        if k == 0:
            print(f"{k:>4} | {x[0]: .6f} {x[1]: .6f} {fx: .6f}      ----")
        else:
            print(f"{k:>4} | {x[0]: .6f} {x[1]: .6f} {fx: .6f}  {alphas[k-1]: .6f}")

    # ------------------------------------------------------------------
    # Plot contours and the G.D. trajectory
    xs = np.linspace(-2, 2, 400)
    ys = np.linspace(-1, 3, 400)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X) ** 2 + 100 * (Y - X ** 2) ** 2

    plt.figure(figsize=(6, 5))
    cs = plt.contour(X, Y, Z, levels=40, linewidths=0.5)
    plt.clabel(cs, inline=True, fontsize=6, fmt="%.0f")
    path_arr = np.array(path)
    plt.plot(path_arr[:, 0], path_arr[:, 1], marker="o", linewidth=1.5, markersize=5)
    plt.title("Gradient Descent with Armijo Line‑Search (3 steps)")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
