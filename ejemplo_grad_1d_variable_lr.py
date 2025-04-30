import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función a minimizar
def f(x):
    return (x - 3)**2 + 2

# Derivada de la función
def df(x):
    return 2*(x - 3)

# Gradiente descendiente con tasa de aprendizaje variable
def gradient_descent_var(x0, tol, max_iter):
    iter_data = []
    x = x0
    for i in range(1, max_iter+1):
        grad = df(x)
        lr = 0.25 / np.sqrt(i)
        iter_data.append((i-1, x, f(x), grad, lr))
        if abs(grad) < tol:
            break
        x -= lr * grad
    return iter_data

# Parámetros
x0 = 0
tolerance = 0.0001
max_iterations = 50

# Ejecutar el método
iterations_var = gradient_descent_var(x0, tolerance, max_iterations)

# Crear tabla
df_var = pd.DataFrame(iterations_var, columns=['Iteration', 'x', 'f(x)', 'df(x)', 'learning rate'])
print(df_var)

# Gráfica
x_values = np.linspace(-1, 5, 400)
f_values = f(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, f_values, label='f(x)')
plt.plot(df_var['x'], df_var['f(x)'], 'bo-', label='Iteraciones')
plt.title('Gradiente Descendiente (Learning Rate Variable)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.show()

