import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def find_circle(points):
    # Находим центр масс точек
    center_mass = np.mean(points, axis=0)
    
    # Находим максимальное расстояние от центра масс до каждой точки
    max_distance = np.max(np.linalg.norm(points - center_mass, axis=1))
    center_mass = np.append(center_mass, 0)
    return center_mass, 2*max_distance

def to_3d(points_2d, center_mass, max_distance):
    points_3d = []
    for point in points_2d:
        # Вычисляем z-координату для каждой точки
        z = max_distance - np.sqrt(max_distance**2 - (point[0] - center_mass[0])**2 - (point[1] - center_mass[1])**2)
        points_3d.append([point[0], point[1], z])
    
    return np.array(points_3d)

def visualize_3d(points_3d, center_mass, max_distance):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Визуализация исходных точек
    ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2], c='b', label='Полученные точки')
    ax.scatter(points_3d[:,0], points_3d[:,1], 0, c='y', label='Исходные точки')

    # Визуализация проекций на сферу
    ax.scatter(center_mass[0], center_mass[1], center_mass[2], c='r', label='Центр масс')

    # Визуализация сферы
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = center_mass[0] + max_distance * np.outer(np.cos(u), np.sin(v))
    y = center_mass[1] + max_distance * np.outer(np.sin(u), np.sin(v))
    z = max_distance + center_mass[2] + max_distance * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='gray', alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.legend()
    plt.show()

# Пример использования
points_2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8],[-2,-2]])
center_mass, max_distance = find_circle(points_2d)
points_3d = to_3d(points_2d, center_mass, max_distance)
visualize_3d(points_3d, center_mass, max_distance)