import numpy as np


def simulate_ising_with_vacancies(T, L, vacancy_prob):
    """
    Simulates the 2D Ising model with vacancies for a given temperature, lattice size, and vacancy probability.

    Args:
        T (float): Temperature in units of J/kb.
        L (int): Lattice size.
        vacancy_prob (float): Probability of a site being vacant.

    Returns:
        energy (float): Average energy per spin.
        mag (float): Average magnetization per spin.
        corr_len (float): Correlation length.
    """

    # Initialize the lattice with random spins and vacancies
    lattice = np.random.choice([-1, 1], size=(L, L))
    vacancies = np.random.choice([0, 1], size=(L, L), p=[1 - vacancy_prob, vacancy_prob])
    lattice *= vacancies

    # Define the Boltzmann constant and the interaction strength J
    kb = 1
    J = 1

    # Define a function to calculate the energy change for flipping a spin
    def deltaE(lattice, i, j):
        neighbors = lattice[(i - 1) % L, j] + lattice[(i + 1) % L, j] + lattice[i, (j - 1) % L] + lattice[
            i, (j + 1) % L]
        return 2 * J * lattice[i, j] * neighbors

    # Define a function to calculate the magnetization and energy of the lattice
    def calcMagE(lattice):
        mag = np.sum(lattice)
        energy = 0
        for i in range(L):
            for j in range(L):
                energy -= J * lattice[i, j] * (lattice[(i - 1) % L, j] + lattice[i, (j - 1) % L])
        return mag / (L * L), energy / (L * L)

    # Run the simulation for a fixed number of Monte Carlo steps
    nsteps = 10000
    mags = np.zeros(nsteps)
    energy = np.zeros(nsteps)
    for step in range(nsteps):
        # Choose a random spin to flip
        i = np.random.randint(L)
        j = np.random.randint(L)

        # Calculate the energy change for flipping the spin
        dE = deltaE(lattice, i, j)

        # Accept or reject the move based on the Metropolis criterion
        if dE <= 0 or np.exp(-dE / (kb * T)) > np.random.rand():
            lattice[i, j] *= -1

        # Calculate the magnetization and energy after the move
        mag, ene = calcMagE(lattice)

        # Save the magnetization and energy at this step
        mags[step] = mag
        energy[step] = ene

    # Calculate the correlation length
    corr_len = np.correlate(mags - np.mean(mags), mags - np.mean(mags), mode='full')[nsteps - 1:]
    corr_len /= corr_len[0]
    xi = np.sum(corr_len)

    return np.mean(energy), np.mean(mags), xi
T = 2.26
L = 1000
vacancy_prob = 0.1

energy, mag, corr_len = simulate_ising_with_vacancies(T, L, vacancy_prob)

print(corr_len)
