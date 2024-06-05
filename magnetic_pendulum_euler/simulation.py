import numpy as np

def run_simulation(pendulum, dt):
    trajectory = []  # Initialise blank list
    while True:
        pos = pendulum.update(dt)  # Update the pendulum position
        trajectory.append(pos.copy())  # Store the current position
        if np.linalg.norm(pendulum.vel) < 1e-3:
            break  # Stop if pendulum comes to rest
    return np.array(trajectory)  # Return as numpy array