
"""
dogleg_2d_full.py  –  Método dog–leg / trust‑region en 2‑D
==========================================================
* Ejecuta y grafica automáticamente la trayectoria y las regiones de confianza.
* Uso rápido:  python dogleg_2d_full.py
"""
import numpy as np
import matplotlib.pyplot as plt

# --------- Función cuadrática de prueba ----------------------------------
H = np.array([[3.0, 1.0],
              [1.0, 2.0]])
c = np.array([-2.0, -6.0])

def f(x):
    return 0.5 * x @ H @ x + c @ x

def grad(x):
    return H @ x + c

def hess(x):
    return H

# --------- Algoritmo dog‑leg / trust‑region ------------------------------
def dogleg_2d(x0, delta0=1.0, delta_max=5.0,
              eta1=0.25, eta2=0.75,
              tol=1e-6, max_iter=20):
    x = np.asarray(x0, float)
    delta = float(delta0)
    path = []                # [(x_k, delta_k), …]
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < tol:
            path.append((x.copy(), delta)); break

        B = hess(x)
        pB = -np.linalg.solve(B, g)           # Paso de Newton
        gBg = g @ B @ g
        alpha_sd = (g @ g) / gBg if gBg > 0 else delta/np.linalg.norm(g)
        pU = -alpha_sd * g                    # Punto de Cauchy

        if np.linalg.norm(pB) <= delta:
            p = pB
        elif np.linalg.norm(pU) >= delta:
            p = (delta / np.linalg.norm(pU)) * pU
        else:
            diff = pB - pU
            a = diff @ diff
            b = 2 * (pU @ diff)
            cquad = pU @ pU - delta**2
            tau = (-b + np.sqrt(b**2 - 4*a*cquad)) / (2*a)
            p = pU + tau * diff

        rho = (f(x) - f(x+p)) / -(g @ p + 0.5*p @ B @ p)
        if rho < eta1:
            delta *= 0.5                          # Rechazamos paso
        else:
            x = x + p                             # Aceptamos
            if rho > eta2:
                delta = min(2*delta, delta_max)

        path.append((x.copy(), delta))
        if np.linalg.norm(g) < tol:
            break
    return x, path

# ------------------ Bloque de ejecución / visualización ------------------
if __name__ == "__main__":
    x0 = np.array([0.0, 0.0])
    x_opt, path = dogleg_2d(x0, delta0=1.0)

    # Contornos para la gráfica
    xx, yy = np.meshgrid(np.linspace(-2, 4, 400),
                         np.linspace(-4, 4, 400))
    zz = 0.5 * (H[0,0]*xx**2 + 2*H[0,1]*xx*yy + H[1,1]*yy**2) + \
         c[0]*xx + c[1]*yy

    plt.figure(figsize=(7,7))
    plt.contour(xx, yy, zz, levels=40, cmap='viridis')

    # Dibujar círculos y flechas
    for i, (xk, delt) in enumerate(path):
        th = np.linspace(0, 2*np.pi, 200)
        plt.plot(xk[0] + delt*np.cos(th),
                 xk[1] + delt*np.sin(th),
                 color='orange' if i==len(path)-1 else 'lightcoral',
                 alpha=0.6)
        plt.scatter(xk[0], xk[1],
                    color='green' if i==len(path)-1 else 'red', zorder=5)
        if i > 0:
            x_prev, _ = path[i-1]
            plt.arrow(x_prev[0], x_prev[1],
                      xk[0]-x_prev[0], xk[1]-x_prev[1],
                      length_includes_head=True, head_width=0.08,
                      color='black', lw=1.2)

    plt.title("Dog‑leg / trust‑region en 2‑D")
    plt.xlabel("$x_1$"); plt.ylabel("$x_2$")
    plt.axis('equal'); plt.grid(True); plt.tight_layout()
    plt.show()

    print("Optimum aproximado:", x_opt)
