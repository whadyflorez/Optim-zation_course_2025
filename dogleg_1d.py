
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (x - 2)**4 + (x - 2)**2 + 1

def df(x):
    return 4*(x - 2)**3 + 2*(x - 2) 

def d2f(x):
    return 12*(x - 2)**2 + 2

def dogleg_1d(x0, delta0=1.0, delta_max=5.0, eta1=0.25, eta2=0.75,
              tol=1e-6, max_iter=20):
    x, delta = float(x0), float(delta0)
    hist = []
    for k in range(max_iter):
        g, B = df(x), d2f(x)
        if abs(g) < tol:
            hist.append((x, delta)); break
        pB = -g / B
        p = pB if abs(pB) <= delta else np.sign(pB)*delta
        m0 = f(x); mp = m0 + g*p + 0.5*B*p**2
        rho = (m0 - f(x+p)) / (m0 - mp)
        if rho < eta1:
            delta *= 0.5
        else:
            x += p
            if rho > eta2 and abs(p) > 0.9*delta:
                delta = min(2*delta, delta_max)
        hist.append((x, delta))
        if abs(g) < tol: break
    return x, hist


x_opt, hist = dogleg_1d(-1.5, 1.0)
xs, ds = zip(*hist)
xp = np.linspace(-2, 5, 500)
yp = [f(xi) for xi in xp]
plt.plot(xp, yp)
plt.errorbar(xs, [f(xi) for xi in xs], xerr=ds, fmt='ro', capsize=4)
plt.show()
print("Optimal x:", x_opt)
