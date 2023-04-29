import numpy as np

# Ising model simulation parameters
L = 40  # lattice size
N = L*L  # number of spins
n_temps = 20  # number of temperatures to simulate
temps = np.linspace(0.5, 4.0, n_temps)

# Initialize random spin configuration
spins = np.random.choice([-1, 1], size=(L, L))

# Define function to calculate energy difference for flipping spin at (i,j)
def delta_E(spins, i, j):
    return 2*spins[i,j]*(spins[(i+1)%L,j] + spins[i,(j+1)%L] + spins[(i-1)%L,j] + spins[i,(j-1)%L])

# Define function for Monte Carlo simulation
def ising_monte_carlo(temperature, n_steps):
    beta = 1/temperature
    energy = 0
    mag = np.sum(spins)
    for step in range(n_steps):
        i, j = np.random.randint(L), np.random.randint(L)
        dE = delta_E(spins, i, j)
        if dE <= 0 or np.random.rand() < np.exp(-beta*dE):
            spins[i,j] *= -1
            energy += dE
            mag += 2*spins[i,j]
    energy_density = energy/N
    mag_density = mag/N
    return energy_density, mag_density

# Generate and save data for each temperature
for temp in temps:
    energy_densities = []
    mag_densities = []
    for i in range(10000):
        ed, md = ising_monte_carlo(temp, 100)
        energy_densities.append(ed)
        mag_densities.append(md)
    filename = f"ising_data_{temp:.2f}.npz"
    np.savez(filename, inputs=np.array(mag_densities), targets=np.array(energy_densities))
