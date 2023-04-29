import numpy as np

# Define the lattice size
L = 10

# Define the temperature and coupling constant
T = 400.0
J = 1.0

# Define the number of Monte Carlo steps
nsteps = 10000

# Define the lattice and initial spin configuration
lattice = np.ones((L, L))
spin_config = np.random.choice([-1, 1], size=(L, L))

# Define the energy function
def calculate_energy(lattice, J):
    energy = 0
    for i in range(L):
        for j in range(L):
            energy += -J * lattice[i][j] * (lattice[(i+1)%L][j] + lattice[i][(j+1)%L])
    return energy

# Define the Metropolis algorithm
def metropolis(lattice, T, J):
    for i in range(L):
        for j in range(L):
            # Choose a random spin to flip
            x = np.random.randint(0, L)
            y = np.random.randint(0, L)

            # Calculate the change in energy
            dE = 2 * J * lattice[x][y] * (lattice[(x+1)%L][y] + lattice[(x-1)%L][y] +
                                          lattice[x][(y+1)%L] + lattice[x][(y-1)%L])

            # If the change in energy is negative, flip the spin
            if dE < 0:
                lattice[x][y] = -lattice[x][y]
            # If the change in energy is positive, flip the spin with probability e^(-dE/T)
            elif np.exp(-dE/T) > np.random.rand():
                lattice[x][y] = -lattice[x][y]

    return lattice

# Run the simulation for nsteps Monte Carlo steps
spin_configs = []
for i in range(nsteps):
    spin_config = metropolis(spin_config, T, J)
    spin_configs.append(spin_config.reshape(1, L*L))

# Save the spin configurations as a dataset
np.savetxt('ising_data.csv', np.concatenate(spin_configs, axis=0), delimiter=',')

# Load the CSV file
spin_configs = np.loadtxt('ising_data.csv', delimiter=',')

# Print the shape of the array
print(spin_configs.shape)

# Print the first 10 rows of the array
print(spin_configs[:10])
