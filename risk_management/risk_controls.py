# =========================
# Risk Controls (FINAL)
# =========================

def apply_stop_loss(env, current_price, stop_loss_pct=0.05):
    """
    Trigger stop loss if price drops below threshold.
    """

    if env.shares == 0:
        return False

    # approximate entry price
    entry_price = (env.initial_cash - env.cash) / env.shares if env.shares > 0 else current_price

    if current_price < entry_price * (1 - stop_loss_pct):
        # SELL ALL
        env.cash += env.shares * current_price
        env.shares = 0
        return True

    return False


def apply_position_limit(env, max_position_pct=0.8):
    """
    Penalize if too much capital is invested.
    """

    total_value = env.cash + env.shares * env.prices[env.current_step]

    if total_value == 0:
        return 0

    position_value = env.shares * env.prices[env.current_step]

    position_ratio = position_value / total_value

    # penalty if too large
    if position_ratio > max_position_pct:
        return -2  # penalty

    return 0