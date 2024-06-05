import numpy as np
from scipy.integrate import solve_ivp

def run_simulation(pendulum, dt, vel_threshold=1e-3):
    t_span = (0, 1e6)  # Arbitrarily large end time
    t_eval = [0]  # Start with initial time only
    initial_conditions = pendulum.initial_conditions()

    def event_stationary(t, y):
        """Event to stop integration when the bob is stationary."""
        return np.linalg.norm(y[2:]) - vel_threshold

    event_stationary.terminal = True
    event_stationary.direction = -1

    # Use solve_ivp with RK45 method and an event to stop integration
    solution = solve_ivp(
        pendulum._total_force,
        t_span,
        initial_conditions,
        method='RK45',
        events=event_stationary,
        dense_output=True,
        max_step=dt
    )

    if not solution.success:
        raise RuntimeError('Integration failed: ' + solution.message)

    # Extract the positions from the solution at required intervals
    t_final = solution.t_events[0][0] if solution.t_events[0].size > 0 else t_span[1]
    t_eval = np.arange(0, t_final, dt)
    trajectory = solution.sol(t_eval)[:2].T
    return trajectory
