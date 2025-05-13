import numpy as np
import matplotlib.pyplot as plt

# Crear malla para la función f(x, y)
x = np.linspace(0, 250, 400)
y = np.linspace(0, 500, 400)
X, Y = np.meshgrid(x, y)
Z = 50 * X + 100 * Y

# Crear figura y eje 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Graficar la superficie de f(x, y)
ax.plot_surface(X, Y, Z, alpha=0.6, cmap='viridis', edgecolor='none')

# Agregar curvas de nivel proyectadas sobre el plano z = 0
ax.contour(X, Y, Z, levels=10, cmap='inferno', linestyles="--", offset=0)

# Graficar las restricciones como funciones normales sobre el plano z = 0
x_vals = np.linspace(0, 250, 500)
y_line1 = (2500 - 10 * x_vals) / 5
y_line2 = (2000 - 4 * x_vals) / 10
y_line3 = (450 - x_vals) / 1.5

# Dibujar las líneas sobre el plano xy con z = 0
ax.plot(x_vals, y_line1, 0 * x_vals, color='red', linewidth=3, label='10x + 5y = 2500')
ax.plot(x_vals, y_line2, 0 * x_vals, color='blue', linewidth=3, label='4x + 10y = 2000')
ax.plot(x_vals, y_line3, 0 * x_vals, color='green', linewidth=3, label='x + 1.5y = 450')

# Configurar la vista, etiquetas y leyenda
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x, y) = 50x + 100y')
ax.set_title('Superficie de f(x, y) y restricciones proyectadas sobre el plano xy')
ax.view_init(elev=17.55, azim=-55.56)
ax.legend()

plt.tight_layout()
plt.show()
