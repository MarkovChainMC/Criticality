import numpy as np
import os


def ising_data(L=16, num_T=20, T_range=[0.5, 4.0], num_samples=1000):
    """
    Generates Ising model data for a range of temperatures and saves each dataset to a separate file with the temperature in the filename.

    Parameters:
        L (int): The size of the lattice
        num_T (int): The number of temperatures to generate data for
        T_range (tuple): The range of temperatures to generate data for
        num_samples (int): The number of samples to generate for each temperature

    Returns:
        None
    """
    for i, T in enumerate(np.linspace(T_range[0], T_range[1], num_T)):
        # Generate data
        data = []
        for j in range(num_samples):
            # Generate random spin configuration
            spins = np.random.choice([-1, 1], size=(L, L))
            E = -np.sum(spins * np.roll(spins, 1, axis=0) + spins * np.roll(spins, 1, axis=1))
            # Metropolis algorithm
            for k in range(10 * L * L):
                # Choose random spin
                x, y = np.random.randint(0, L, size=2)
                # Calculate energy change
                delta_E = 2 * spins[x, y] * (
                            spins[x, (y + 1) % L] + spins[x, (y - 1) % L] + spins[(x + 1) % L, y] + spins[
                        (x - 1) % L, y])
                # Flip spin with probability given by Boltzmann factor
                if delta_E <= 0 or np.random.rand() < np.exp(-delta_E / T):
                    spins[x, y] = -spins[x, y]
                    E += delta_E
            # Compute magnetization and energy densities
            M = np.sum(spins)
            E_density = E / (L * L)
            M_density = M / (L * L)
            data.append([E_density, M_density])
        # Save data to file
        filename = f'ising_data_T_{T:.3f}.npz'
        np.savez(filename, data=np.array(data))
