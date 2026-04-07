def check_drawdown_protection(
    bankroll_history: list[float],
    current_bankroll: float,
    peak_bankroll:   float,
    halt_threshold:  float = 0.10,  # halt at 10% drawdown from peak
) -> bool:
    """
    Returns True if betting should halt.
    
    IMPORTANT: this function must ACTUALLY prevent bet sizing,
    not just log a warning. The caller must check the return value.
    """
    drawdown = (peak_bankroll - current_bankroll) / peak_bankroll
    return drawdown >= halt_threshold