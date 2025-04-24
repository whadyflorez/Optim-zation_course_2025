import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Función a minimizar (ejemplo del libro)
def f(lam):
    return lam**5 - 5*lam**3 - 20*lam + 5

# Derivada de la función
def df(lam):
    return 5*lam**4 - 15*lam**2 - 20

# Método de interpolación cúbica
def cubic_interpolation(a, b, tol, max_iter):
    iter_data = []
    for i in range(max_iter):
        fa, fb = f(a), f(b)
        dfa, dfb = df(a), df(b)
        
        Z = (3*(fa - fb)/(b - a)) + dfa + dfb
        Q = np.sqrt(Z**2 - dfa*dfb)
        lam_star = a + (dfa + Z + Q)*(b - a)/(dfa + dfb + 2*Z)
        
        f_star = f(lam_star)
        iter_data.append((i, a, b, lam_star, f_star))
        
        # Criterio de convergencia
        if abs(df(lam_star)) < tol:
            break
        
        # Actualización del intervalo
        if df(lam_star) < 0:
            a = lam_star
        else:
            b = lam_star
    
    return iter_data, lam_star

# Parámetros iniciales
a, b = 0, 3
tolerance = 0.001
max_iterations = 15

# Ejecutar el método
iterations, minimum = cubic_interpolation(a, b, tolerance, max_iterations)

# Crear tabla de iteraciones
df_iter = pd.DataFrame(iterations, columns=['Iteration', 'a', 'b', 'lambda*', 'f(lambda*)'])
print(df_iter)

# Gráfica de la función y puntos de iteración
lambda_values = np.linspace(-0.5, 2.5, 400)
f_values = f(lambda_values)

plt.figure(figsize=(10, 6))
plt.plot(lambda_values, f_values, label='f(λ)')
plt.plot(df_iter['lambda*'], df_iter['f(lambda*)'], 'ro-', label='Iteraciones')

for i, (lam_star, f_star) in enumerate(zip(df_iter['lambda*'], df_iter['f(lambda*)'])):
    plt.text(lam_star, f_star, f'{i}', fontsize=9, ha='right')

plt.xlabel('λ')
plt.ylabel('f(λ)')
plt.title('Minimización usando método de interpolación cúbica')
plt.legend()
plt.grid(True)
plt.show()