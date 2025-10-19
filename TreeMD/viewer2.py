import matplotlib.pyplot as plt
import numpy as np
import core

# --- Параметры симуляции ---
sim = core.Simulation(
    N=60,
    box_size=1.0,
    vel_init=0.1,
    dt=1e-6,
    eps=0.1,
    sigma=0.01,
    k=7.0,
    G=0.001,
    use_gravity=False
)

# --- Настройка визуализации ---
#plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 1.0)  # используем фиксированный box_size
ax.set_ylim(0, 1.0)
points, = ax.plot([], [], 'o', ms=4)
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# --- Хранение энергии для графика ---
energy_history = []

# --- Основной цикл ---
for frame in range(1000):
    stats = sim.step(1000)
    energy = stats['E_total']
    energy_history.append(energy)
    pos = np.array(sim.get_positions())

    points.set_data(pos[:, 0], pos[:, 1])
    text.set_text(f"Frame {frame:04d}, E = {energy:.4e}")
    plt.pause(0.001)

plt.ioff()  # выключаем интерактивный режим
plt.show()

# --- График энергии ---
plt.figure(figsize=(6,4))
plt.plot(energy_history, label='E_total')
plt.xlabel('Simulation step')
plt.ylabel('Total energy')
plt.title('Energy vs Simulation Step')
plt.legend()
plt.grid(True)
plt.show()

input("Press Enter to exit\n")
