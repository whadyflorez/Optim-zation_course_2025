import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función a minimizar
def f(x):
    return (x - 3)**2 + 2

# Derivada de la función
def df(x):
    return 2*(x - 3)

# Gradiente descendiente con tasa de aprendizaje constante
def gradient_descent_const(lr, x0, tol, max_iter):
    iter_data = []
    x = x0
    for i in range(max_iter):
        grad = df(x)
        iter_data.append((i, x, f(x), grad))
        if abs(grad) < tol:
            break
        x -= lr * grad
    return iter_data

# Parámetros
x0 = 0
lr_const = 0.25
tolerance = 0.0001
max_iterations = 50

# Ejecutar el método
iterations_const = gradient_descent_const(lr_const, x0, tolerance, max_iterations)

# Crear tabla
df_const = pd.DataFrame(iterations_const, columns=['Iteration', 'x', 'f(x)', 'df(x)'])
print(df_const)

# Gráfica
x_values = np.linspace(-1, 5, 400)
f_values = f(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, f_values, label='f(x)')
plt.plot(df_const['x'], df_const['f(x)'], 'ro-', label='Iteraciones')
plt.title('Gradiente Descendiente (Learning Rate Constante)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

