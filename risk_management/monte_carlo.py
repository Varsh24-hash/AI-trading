import numpy as np
import matplotlib.pyplot as plt


# =========================
# 🎲 Monte Carlo Simulation
# =========================
def monte_carlo_simulation(returns, num_simulations=100, horizon=50, initial_value=10000):
    """
    Simulate future portfolio values using historical returns.
    """

    returns = np.array(returns)

    simulations = []

    for _ in range(num_simulations):

        simulated_path = [initial_value]

        for _ in range(horizon):

            # sample random return from historical data
            r = np.random.choice(returns)

            new_value = simulated_path[-1] * (1 + r)
            simulated_path.append(new_value)

        simulations.append(simulated_path)

    return np.array(simulations)


# =========================
# 📊 Plot Simulation
# =========================
def plot_simulation(simulations):
    """
    Plot multiple simulated paths.
    """

    plt.figure(figsize=(10, 6))

    for sim in simulations:
        plt.plot(sim, alpha=0.3)

    plt.title("Monte Carlo Simulation of Portfolio")
    plt.xlabel("Time Steps")
    plt.ylabel("Portfolio Value")
    plt.grid()

    plt.show()


# =========================
# 📉 Risk Metrics
# =========================
def compute_var(simulations, confidence_level=0.95):
    """
    Compute Value at Risk (VaR)
    """

    final_values = simulations[:, -1]

    var = np.percentile(final_values, (1 - confidence_level) * 100)

    return var


def compute_cvar(simulations, confidence_level=0.95):
    """
    Compute Conditional VaR (Expected Shortfall)
    """

    final_values = simulations[:, -1]

    var = compute_var(simulations, confidence_level)

    cvar = final_values[final_values <= var].mean()

    return cvar