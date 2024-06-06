import numpy as np
from scipy.integrate import solve_ivp

def run_simulation(pendulum, dt, tmax):
    t_span = (0, tmax)  # Arbitrarily large end time
    t_eval = np.arange(0, tmax, dt)  # Start with initial time only
    initial_conditions = pendulum.initial_conditions()

    # Use solve_ivp with RK45 method and an event to stop integration
    solution = solve_ivp(
        pendulum._total_force,
        t_span,
        initial_conditions,
        method='RK45',
        t_eval=t_eval
    )

    if not solution.success:
        raise RuntimeError('Integration failed: ' + solution.message)

    trajectory = solution.y[:2].T
    return trajectory
