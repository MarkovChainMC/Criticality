import numpy as np

def monte_carlo_ising(L, N, T_min=1.5, T_max=3.0, T_step=0.1):
    # calculate the temperature range
    T_range = np.arange(T_min, T_max + T_step, T_step)

    # initialize the dataset
    data = []
    labels = []

    # loop over the temperature range
    for T in T_range:
        # initialize the spin configuration randomly
        spin_config = np.random.choice([-1, 1], size=(L, L))

        # perform N Monte Carlo steps at the current temperature
        for n in range(N):
            # select a random spin
            i, j = np.random.randint(0, L, size=2)

            # calculate the change in energy if the spin is flipped
            delta_E = 2 * spin_config[i, j] * (spin_config[(i-1)%L, j] +
                                                spin_config[(i+1)%L, j] +
                                                spin_config[i, (j-1)%L] +
                                                spin_config[i, (j+1)%L])

            # if the energy decreases or the Metropolis criterion is met, flip the spin
            if delta_E < 0 or np.exp(-delta_E/T) > np.random.rand():
                spin_config[i, j] *= -1

        # add the spin configuration and the label to the dataset
        data.append(spin_config.ravel())
        if T < 2.25:
            labels.append(0)  # ordered phase
        elif T > 2.75:
            labels.append(1)  # disordered phase
        else:
            labels.append(2)  # critical phase

        print("Temperature:", T)  # print the current temperature for progress tracking

    # save the dataset to a file
    np.savez("ising_data.npz", data=np.array(data), labels=np.array(labels))

    return data, labels

# example usage
data, labels = monte_carlo_ising(L=10, N=10000, T_min=1, T_max=2.5, T_step=0.1)
