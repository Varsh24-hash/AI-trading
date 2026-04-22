import numpy as np

def calculate_var(values, confidence=0.95):
    """
    Calculate Value at Risk (VaR) for a given portfolio based on historical returns.

    Args:
    values (list): Portfolio values over time.
    confidence (float): Confidence level (e.g., 0.95 for 95% confidence).

    Returns:
    float: Value at Risk (VaR).
    """
    returns = np.diff(values) / values[:-1]
    return np.percentile(returns, (1 - confidence) * 100)