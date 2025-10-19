import core, numpy as np, matplotlib.pyplot as plt

sim = core.Simulation(N=30, box_size=1.0, G=.001, m=1.0, dt=0.00005, u=0.1)

E_total = []
for i in range(500000):
    sim.step()
    if i % 100 == 0:
        Etot, Ek, Ep = sim.get_energy()
        E_total.append(Etot)

plt.plot(E_total)
plt.xlabel('step')
plt.ylabel('Total energy')
plt.title('Energy conservation test')
plt.show()
