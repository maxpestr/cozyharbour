import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "build"))
import core
import numpy as np
import matplotlib.pyplot as plt
import time

N = 100
sim = core.Simulation(N, 1.0)

plt.ion()
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
points, = ax.plot([], [], 'o', ms=4)

for frame in range(5000):
    for _ in range(5):  # несколько шагов между кадрами
        sim.step()
    pos = sim.get_positions()
    points.set_data(pos[:,0], pos[:,1])
    plt.pause(0.01)
