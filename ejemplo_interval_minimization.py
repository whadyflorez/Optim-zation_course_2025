import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Definir la función a minimizar
def f(x):
    return x**2 - 4*x + 4

# Implementación del método Interval Halving
def interval_halving(a, b, tol, max_iter):
    iter_data = []
    iter_count = 0
    while (b - a) / 2 > tol and iter_count < max_iter:
        xm = (a + b) / 2
        L = b - a
        x1 = a + L / 4
        x2 = b - L / 4
        f1, f2, fm = f(x1), f(x2), f(xm)
        
        iter_data.append((iter_count, a, b, xm, f(xm)))

        if f1 < fm:
            b = xm
            xm = x1
        elif f2 < fm:
            a = xm
            xm = x2
        else:
            a = x1
            b = x2
        
        iter_count += 1
    
    return iter_data, xm

# Parámetros iniciales
a, b = 0, 3
tolerance = 0.001
max_iterations = 20

# Ejecutar el método
iterations, minimum = interval_halving(a, b, tolerance, max_iterations)

# Crear y mostrar tabla de iteraciones
df = pd.DataFrame(iterations, columns=['Iteration', 'a', 'b', 'xm', 'f(xm)'])
print(df)

# Gráfica de la función y las iteraciones
x_values = np.linspace(0, 3, 400)
y_values = f(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='f(x)')
for i, (_, a, b, xm, _) in enumerate(iterations):
    plt.plot(xm, f(xm), 'o', label=f'Iteración {i}' if i < 5 else "", markersize=6)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Minimización usando el Método Interval Halving')
plt.legend()
plt.grid(True)
plt.show()
