import matplotlib.pyplot as plt
import numpy as np
import core

# --- Параметры симуляции ---
box_size = 1.0
vel_init = 0.1
sim = core.Simulation(
    N=60,
    box_size=box_size,
    vel_init=vel_init,
    dt=1e-4,
    eps=0.1,
    sigma=0.01,
    k=5.0,
    G=0.001,
    use_gravity=False
)

# --- Настройка визуализации ---
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
points, = ax.plot([], [], 'o', ms=4)
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# --- Основной цикл ---
for frame in range(1000):
    stats = sim.step(200)          # один шаг симуляции
    energy = stats['E_total']      # получаем энергию
    pos = np.array(sim.get_positions())

    points.set_data(pos[:, 0], pos[:, 1])
    text.set_text(f"Frame {frame:04d}, E = {energy:.4e}")
    plt.pause(0.01)

input("Press Enter to exit\n")
